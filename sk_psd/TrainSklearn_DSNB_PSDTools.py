# -*- coding:utf-8 -*-
# @Time: 2022/1/5 10:48
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: TrainSklearn_DSNB_PSDTools.py
import os.path

import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
from LoadMultiFiles import LoadOneFileUproot, LoadFileListUprootOptimized
from HistTools import GetBinCenter
from PlotDetectorGeometry import GetR3

import pickle
from NumpyTools import Replace,AlignEvents,SplitEventsDict,ShuffleDataset, MergeEventsDictionary

class TrainTool:
    def __init__(self):
        # Set Dataset Variables
        self.dir_PSD_diff_particle = {}
        self.input_train = []
        self.target_train = []
        self.map_tag_particles = {"Atm":0, "DSNB":1}
        self.key_tag = "tag"
        self.ratio_split_for_train = 0.5

        ## Get Bins Setting
        self.bins = np.loadtxt("/afs/ihep.ac.cn/users/l/luoxj/PSD_LoweE/alpha/Bins_Setting.txt",delimiter=",", dtype=float)
        self.bins_center = GetBinCenter(self.bins)
        self.bins_width = np.diff(self.bins)

        ## Set Filter For Events
        self.filter = {"Eqe": [10, 30], "r3": [0, 4096]}

        # Training Settings
        self.normalized_input=False
        self.strategy = "Combine"
        self.path_model = "./model_PSDTools/"
        self.name_file_model = self.path_model+"model_atm_"+self.strategy+".pkl"
        self.path_save_fig = "./figure/"

        if not os.path.exists(self.path_save_fig):
            os.mkdir(self.path_save_fig)

    def SetTrainingStrategy(self, strategy):
        # strategy options:
        self.strategy = strategy


    def LoadDataset(self, dir_path:dict):
        # Prepare Files List
        list_corresponding_keys = []
        file_list = []
        for key, path in dir_path.items():
            list_corresponding_keys.append(key)
            file_list += [path]
        # dir_n_files_to_load = {"alpha":300, "e-":300}
        # for key, tag in self.map_tag_particles.items():
        #     list_corresponding_keys += [key]*dir_n_files_to_load[key]
        #     file_list += [f"root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/PSD_LowE/{key}/PSD/root/PSD-{i}.root" for i in range(dir_n_files_to_load[key])]

        # Load Files
        self.dir_PSD_diff_particle = LoadFileListUprootOptimized(file_list,list_corresponding_keys=list_corresponding_keys,
                                    name_branch="TMVAinput", use_multiprocess=True)

        # Append Tags and R3 for the dataset
        for key,dir_PSD in self.dir_PSD_diff_particle.items():
            v_keys = list(dir_PSD.keys())
            self.dir_PSD_diff_particle[key][self.key_tag] = np.array([self.map_tag_particles[key]]*len(dir_PSD[v_keys[0]]))
            # self.dir_PSD_diff_particle[key]["r3"] = np.sum( (dir_PSD["XYZ"]/1000)**2,axis=1)**(3/2)

        # Append additional key in DSNB dataset
        if "Atm" in list(self.dir_PSD_diff_particle.keys()) and "isoz" in list(self.dir_PSD_diff_particle["Atm"].keys()):
            self.dir_PSD_diff_particle["DSNB"]["isoz"] = np.zeros(len(self.dir_PSD_diff_particle["DSNB"][self.key_tag]))
            self.dir_PSD_diff_particle["DSNB"]["ison"] = np.zeros(len(self.dir_PSD_diff_particle["DSNB"][self.key_tag]))
            self.dir_PSD_diff_particle["DSNB"]["id_tag"] = np.zeros(len(self.dir_PSD_diff_particle["DSNB"][self.key_tag]))

        print(self.dir_PSD_diff_particle.keys())

    def NormTimeProfile(self,h_time):
        # return (h_time/self.bins_width) / np.max(h_time/self.bins_width)
        return h_time / np.max(h_time)


    def ConcatenateInput(self, dir_dataset, max_R=17.5, max_E=100):
        input_concatenated = []
        for i in range(len(dir_dataset["Eqe"])):
            # input_concatenated.append(np.concatenate((self.NormTimeProfile(dir_dataset["h_time_without_charge"][i]),
            #                                         self.NormTimeProfile(dir_dataset["h_time_with_charge"][i]),
            #                                         [dir_dataset["r3"][i]/max_R**3], [dir_dataset["Eqe"][i]/max_E] )) )
            input_concatenated.append(np.concatenate((self.NormTimeProfile(dir_dataset["h_time_without_charge"][i]),
                                                  self.NormTimeProfile(dir_dataset["h_time_with_charge"][i]),
                                                  [dir_dataset["r3"][i] / max_R ** 3])))
        return np.array(input_concatenated)

    def FilterForEvents(self, dir_events:dict,dir_filter=None):
        if dir_filter is None:
            dir_filter = {"equen": [0, 1.5], "r3": [0, 4096]} # For example
        v_index = []
        for key in dir_filter.keys():
            v_index.append( (dir_events[key]>=dir_filter[key][0])&
                            (dir_events[key]<=dir_filter[key][1]) )

        # Merge index
        index_return = v_index[0]
        for index in v_index[1:]:
            index_return = (index_return & index)

        for key in dir_events.keys():
            dir_events[key] = dir_events[key][index_return]
        return index_return

    def PrepareInput(self):
        # Filter Events within self.filter
        from collections import Counter
        v_keys = list(self.dir_PSD_diff_particle.keys())
        v_dict = [self.dir_PSD_diff_particle[v_keys[0]], self.dir_PSD_diff_particle[v_keys[1]]]
        for dir in v_dict:
            self.FilterForEvents(dir, self.filter)

        # Align Events Entries
        (self.n_events,self.dir_cut_off) = AlignEvents(v_dict)
        print("dir_cut_off:\t",Counter(self.dir_cut_off[self.key_tag]))

        # Split Dataset for Training and Testing
        v_dict_train_and_test = []
        for dir in v_dict:
            v_dict_train_and_test.append( SplitEventsDict(dir,self.ratio_split_for_train) )

        # Merge signal and background dataset into one dictionary
        self.dir_dataset_train = MergeEventsDictionary([dir_train_test[0] for dir_train_test in v_dict_train_and_test])
        self.dir_dataset_test = MergeEventsDictionary([dir_train_test[1] for dir_train_test in v_dict_train_and_test]+[self.dir_cut_off])

        # Shuffle Dataset
        ShuffleDataset(self.dir_dataset_train)

        # Concatenate ingredient into final input
        self.input_train = self.ConcatenateInput(self.dir_dataset_train)
        self.input_test = self.ConcatenateInput(self.dir_dataset_test)
        self.target_train = self.dir_dataset_train[self.key_tag]
        self.target_test = self.dir_dataset_test[self.key_tag]

        # Clean Unused Dataset to Save memory
        self.dir_dataset_train.clear()
        # self.dir_dataset_test.clear()
        self.dir_cut_off.clear()
        self.dir_PSD_diff_particle.clear()

        # Check Input
        print("Input:\t",self.input_train[0])
        print("Label (Train):\t",Counter(self.target_train))
        print("Label (Test):\t",Counter(self.target_test))

    def AddModels(self):
        # Add multiple models for comparison, look for the definition\
        # of each model on sklearn documentation
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier
        from sklearn.naive_bayes import GaussianNB

        models = {}

        models["LogisticRegression"]  = LogisticRegression()
        models["SVC"] = SVC()
        models["LinearSVC"] = LinearSVC()
        models["KNeighbors"] = KNeighborsClassifier()
        models["DecisionTree"] = DecisionTreeClassifier()
        models["RandomForest"] = RandomForestClassifier()
        rf2 = RandomForestClassifier(n_estimators=100, criterion='gini',
                                                max_depth=10, random_state=0, max_features=None)
        models["RandomForest2"] = rf2
        models["MLPClassifier"] = MLPClassifier(solver='sgd', random_state=0)
        models["GaussianNB"] =  GaussianNB()
        self.models = models

    def TrainModel(self, modelname, pkl_filename=None):
        self.AddModels()

        if pkl_filename is None:
            pkl_filename = self.name_file_model
        if self.normalized_input:
            print("Check normalized status:")
            print(f"input_train mean(should be 0): {self.input_train.mean(axis=0)}")
            print(f"input_train std(should be 1): {self.input_train.std(axis=0)}")
            print(f"input_train shape : {self.input_train.shape}")

        # Train and save a specific model.
        if modelname not in self.models.keys():
            print('Model not supported yet!')
            sys.exit()
        else:
            print('Train model %s' % modelname)
            classifier = self.models[modelname]
            classifier.fit(self.input_train, self.target_train)
            with open(pkl_filename, 'wb') as pfile:
                print("=====>  Saving Model......")
                pickle.dump(classifier, pfile)
                
            if True:
                # predict_proba = classifier.predict_proba(input_test)
                # print(predict_proba.shape)
                print("Score:\t",classifier.score(self.input_test, self.target_test))
                print(classifier.predict(self.input_test[:100]), self.target_test[:100])

    def GetPrediction(self, save_prediction=False):
        with open(self.name_file_model, 'rb') as fr:
            classifier = pickle.load(fr)
        predict_proba = classifier.predict_proba(self.input_test)
        predict_proba_1  = predict_proba[:,1]
        self.dir_dataset_test["PSD"] = np.array(predict_proba_1)
        if save_prediction:
            np.savez(f"{self.path_model}/predict_{self.strategy}.npz", dir_events=self.dir_dataset_test)

    def InterpolateToGetSigEff(self, v_eff_sig, v_eff_bkg, certain_eff_bkg=0.01):
        from scipy.interpolate import interp1d
        v_eff_bkg = np.array(v_eff_bkg)
        v_eff_sig = np.array(v_eff_sig)
        f = interp1d(v_eff_bkg[1:], v_eff_sig[1:], kind="linear")
        eff_sig_return = f(certain_eff_bkg)
        return (certain_eff_bkg, eff_sig_return)

    def GetEfficiencySigma(self, eff):
        relative_sigma_eff = np.sqrt(eff*(1-eff)*len(self.input_test))/len(self.input_test)
        return relative_sigma_eff

    def GetRocCurve(self):
        predict_proba_1 = self.dir_dataset_test["PSD"]
        tag = self.dir_dataset_test["tag"]
        probability_0 = predict_proba_1[tag==0]
        probability_1 = predict_proba_1[tag==1]

        # Get predict probability distribution
        fig1 = plt.figure(self.strategy+"fig")
        ax1 = fig1.add_subplot(111)
        n0, bins0, patches0 = ax1.hist(probability_0, bins=np.linspace(0, 1, 200), color='red', histtype='step',
                                       label='Background')
        n1, bins1, patches1 = ax1.hist(probability_1, bins=np.linspace(0, 1, 200), color='blue', histtype='step',
                                       label='Signal')
        ax1.set_xlim(0, 1)
        plt.semilogy()
        ax1.legend()
        ax1.set_xlabel('Prediction output')
        ax1.set_title(self.strategy)
        fig1.savefig(f"{self.path_save_fig}Predict_Distribution_{self.strategy}.png")

        # Get Roc curve
        eff_bkg = []
        eff_sig = []
        for i in range(len(n0)):
            eff_bkg.append(np.sum(n0[i:]) * 1.0 / np.sum(n0))
            eff_sig.append(np.sum(n1[i:]) * 1.0 / np.sum(n1))

        fig2 = plt.figure("Sig eff. VS Bkg eff.")
        ax2=fig2.add_subplot(111)
        ax2.plot(eff_bkg, eff_sig, label=self.strategy)
        (certain_eff_bkg, eff_sig_return) = self.InterpolateToGetSigEff(v_eff_bkg=eff_bkg, v_eff_sig=eff_sig)
        ax2.scatter(certain_eff_bkg, eff_sig_return, s=20, marker=(5, 1), label=self.strategy)
        print(f"background eff. : {certain_eff_bkg} ---> signal eff. : {eff_sig_return} +- {self.GetEfficiencySigma(eff_sig_return)}")
        ax2.set_xlabel('Background efficiency')
        ax2.set_ylabel('Signal efficiency')
        ax2.set_xlim(0, 0.02)
        ax2.set_ylim(0.8, 1)
        plt.legend()
        fig2.savefig(f"{self.path_save_fig}Efficiency_Sig_and_Bkg_{self.strategy}.png")


if __name__ == '__main__':
    # template_path_data = "/afs/ihep.ac.cn/users/l/luoxj/PSD_LoweE/{}/PSD/root/PSD-*.root"
    template_path_data = "/afs/ihep.ac.cn/users/l/luoxj/junofs_500G/DSNB_data_sklearn/DSNB_IBDSeletion/{}/tmvainput_{}.root"

    train_tool = TrainTool()

    v_wildcard = ["4*", "*[0-4]"]
    dir_path = {}
    for i in range(2):
        key = list(train_tool.map_tag_particles.keys())[i]
        dir_path[key] = template_path_data.format(key, v_wildcard[i])
    # dir_path = {key:template_path_data.format(key) for key in train_tool.map_tag_particles.keys()}
    print(dir_path)
    train_tool.LoadDataset(dir_path)
    train_tool.PrepareInput()
    train_tool.TrainModel("MLPClassifier")

    train_tool.GetPrediction(save_prediction=True)
    train_tool.GetRocCurve()
    #
    # plt.show()


