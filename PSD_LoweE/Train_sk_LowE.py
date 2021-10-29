# -*- coding:utf-8 -*-
# @Time: 2021/10/13 13:28
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: Train_sk_LowE.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys
import os
from pickle import dump,load

import matplotlib.pyplot as plt
import pickle, sys
import random
import pandas as pd
from sklearn import preprocessing

from collections import Counter
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
from LoadMultiFiles import LoadMultiFiles, MergeEventsDictionary
from NumpyTools import Replace,AlignEvents,SplitEventsDict,ShuffleDataset

class SpectrumAna():
    TaskType = ''  # choose a task type from faketest, calibtrain, egammatrain
    dataset = []
    TrainResult = []
    TestRestult = []
    input_train = []
    target_train = []
    input_test = []
    target_test = []
    models = []

    def __init__(self, task, use_coti):
        self.TaskType = task
        self.debug_load_few_files = False
        self.ratio_split_for_train = 0.8
        self.dir_dataset_train = {}
        self.dir_dataset_test = {}
        self.dir_cut_off = {}
        self.normalized_input = False
        self.dir_scheme_map = {0:"without_charge", 1:"with_charge", 2:"combine"}
        self.use_coti = use_coti
        self.option_calib = "_coti" if self.use_coti else ""
        self.path_save_fig = "/afs/ihep.ac.cn/users/l/luoxj/PSD_LoweE/figure/"

        self.path_model = f"./model{self.option_calib}/"
        if not os.path.isdir(self.path_model):
            os.makedirs(self.path_model)

    def SetFilterForEvents(self, dir_filter:dict):
        """

        :param dir_filter: set cut for events for example:{"equen": [0, 1.5], "R3": [0, 4096]}
        :return:
        """
        self.dir_filter = dir_filter

    def FilterForEvents(self, dir_events:dict,dir_filter=None):
        if dir_filter is None:
            dir_filter = {"equen": [0, 1.5], "R3": [0, 4096]} # For example
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

    def PreprocessDataDict(self, v_dict,dir_map_to_replace, key_tag="tag"):
        for dir in v_dict:
            self.FilterForEvents(dir, dir_filter=self.dir_filter)
            print(f"Dataset before aligning:\t",len(dir[key_tag]))

        (self.n_events,self.dir_cut_off) = AlignEvents(v_dict)
        print("dir_cut_off:\t",Counter(self.dir_cut_off[key_tag]))

        v_dict_train_and_test = []
        for dir in v_dict:
            v_dict_train_and_test.append( SplitEventsDict(dir,self.ratio_split_for_train) )

        self.dir_dataset_train = MergeEventsDictionary([dir_train_test[0] for dir_train_test in v_dict_train_and_test])
        self.dir_dataset_test = MergeEventsDictionary([dir_train_test[1] for dir_train_test in v_dict_train_and_test]+[self.dir_cut_off])
        # self.dir_dataset_test = MergeEventsDictionary([dir_train_test[1] for dir_train_test in v_dict_train_and_test])
        # self.dir_dataset_test = MergeEventsDictionary([self.dir_dataset_test, self.dir_cut_off])

        # Substitute tag
        self.dir_dataset_train[key_tag] = Replace(self.dir_dataset_train[key_tag], dir_map_to_replace=dir_map_to_replace)
        self.dir_dataset_test[key_tag] = Replace(self.dir_dataset_test[key_tag], dir_map_to_replace=dir_map_to_replace)

        ShuffleDataset(self.dir_dataset_train)

        # Clear dir_events which will not be used in next part
        self.dir_cut_off = {}
        for dir in v_dict:
            dir = {}
        v_dict_train_and_test = []

        print("Dataset for training:\t",Counter(self.dir_dataset_train[key_tag]))
        print("Dataset for testing:\t",Counter(self.dir_dataset_test[key_tag]))

    def LoadBetaAlphaData(self,filename_alpha:str, filename_e:str, i_scheme:int=0):
        self.strategy = self.dir_scheme_map[i_scheme]
        self.name_file_model = self.path_model+f"{self.strategy}.pkl"

        n_files_to_load = (-1 if not self.debug_load_few_files else 10)
        dir_alpha = LoadMultiFiles(filename_alpha,n_files_to_load=n_files_to_load)
        dir_e = LoadMultiFiles(filename_e, n_files_to_load=n_files_to_load)

        self.PreprocessDataDict([dir_alpha, dir_e], dir_map_to_replace={"alpha":0, "e-":1})

        # Set common input (alias) for module
        if self.strategy == "without_charge":
            self.input_train = self.dir_dataset_train["h_time"]
            self.input_test = self.dir_dataset_test["h_time"]
        elif self.strategy == "with_charge":
            self.input_train = self.dir_dataset_train["h_time_with_charge"]
            self.input_test = self.dir_dataset_test["h_time_with_charge"]
        elif self.strategy == "combine":
            print(np.array(self.dir_dataset_train["h_time"]).shape)
            v_len = []
            for i in range(len(self.dir_dataset_train["h_time"])):
                v_len.append(len(self.dir_dataset_train["h_time"][i]))
            print(Counter(v_len))
            self.input_train = np.concatenate((self.dir_dataset_train["h_time"],
                                               self.dir_dataset_train["h_time_with_charge"]), axis=1)
            self.input_test = np.concatenate((self.dir_dataset_test["h_time"],
                                               self.dir_dataset_test["h_time_with_charge"]), axis=1)
        self.target_train = self.dir_dataset_train["tag"]
        self.target_test = self.dir_dataset_test["tag"]
    
    def AddVertexToInput(self, add_equench=False, max_equen=100):
        if add_equench:
            self.input_train = np.concatenate((self.input_train, self.dir_dataset_train["vertex"]/17.5/1000,
                                               self.dir_dataset_train["equen"].reshape(-1,1)/max_equen), axis=1)
            self.input_test = np.concatenate((self.input_test, self.dir_dataset_test["vertex"]/17.5/1000,
                                              self.dir_dataset_test["equen"].reshape(-1,1)/max_equen), axis=1)
        else:
            self.input_train = np.concatenate((self.input_train, self.dir_dataset_train["vertex"]/17.5/1000), axis=1)
            self.input_test = np.concatenate((self.input_test, self.dir_dataset_test["vertex"]/17.5/1000), axis=1)


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

    def TrainModel(self, modelname, pkl_filename=None ):
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
                pickle.dump(classifier, pfile)
            input_test = self.input_test
            target_test = self.target_test
            if True:
                # predict_proba = classifier.predict_proba(input_test)
                # print(predict_proba.shape)
                print("Score:\t",classifier.score(input_test, target_test))
                print(classifier.predict(input_test[:100]), target_test[:100])

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
        fig1.savefig(f"{self.path_save_fig}Predict_Distribution.png")

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
        fig2.savefig(f"{self.path_save_fig}Efficiency_Sig_and_Bkg.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='sklearn training')
    parser.add_argument("--use_coti", "-c", action="store_true",help="whether use coti waverec else use deconvoluiton waverec" )
    arg = parser.parse_args()
    dir_filter = {"equen": [0, 1.5], "R3": [0, 4096]}
    PSD_tool = SpectrumAna("BetaAlphaTrain", arg.use_coti)
    PSD_tool.SetFilterForEvents(dir_filter)


    PSD_tool.LoadBetaAlphaData(filename_alpha=f"/afs/ihep.ac.cn/users/l/luoxj/PSD_LoweE/Preprocess/dataset_for_train/alpha{PSD_tool.option_calib}/*.npz",
                               filename_e=f"/afs/ihep.ac.cn/users/l/luoxj/PSD_LoweE/Preprocess/dataset_for_train/e-{PSD_tool.option_calib}/*.npz",
                               i_scheme=2)
    PSD_tool.AddVertexToInput(add_equench=True, max_equen=dir_filter["equen"][1])

    PSD_tool.AddModels()
    # PSD_tool.TrainModel('MLPClassifier')
    PSD_tool.GetPrediction(save_prediction=True)
    PSD_tool.GetRocCurve()

    plt.show()
    