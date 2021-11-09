# -*- coding:utf-8 -*-
# @Time: 2021/1/18 21:55
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitLikelihood.py
import numpy as np
import histlite as hl
import matplotlib.pylab as plt
from iminuit import Minuit
from collections import Counter
from tqdm import trange
import random
from matplotlib.colors import LogNorm

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

def SetTitle(title:str, ax:plt.Axes=None):
    """
    # This function is for set title for the 2d histogram of PSD Vs. E distribution
    :param title:
    :param ax:
    :return:
    """
    if ax == None:
        plt.title(title)
        plt.xlabel("Prediction Output")
        plt.ylabel("$E_{quen}$")
    else:
        ax.set_title(title)
        ax.set_xlabel("Prediction Output")
        ax.set_ylabel("$E_{quen}$")
def GetBinCenter(v_edges:np.ndarray):
    return (v_edges[1:]+v_edges[:-1])/2

class FLH:
    """
    This class is for doing the two dimensional fitting(PSD and E) using max Likelihood
    """
    def __init__(self):
        self.check_result = True
        self.v_vertex = np.array([])
        self.v_equen = np.array([])
        self.v_predict = np.array([])
        self.v_R3 = np.array([])
        self.dir_other_bkg = {}
        self.dir_is_FV2 = {}
        self.dir_samples_DSNB_diff_model = {}
        self.dir_samples_NC_diff_model = {}
        self.dir_other_bkg_sys_uncertainty = {}
        self.dir_h2d_other_bkg = {}
        self.dir_h2d_other_bkg_full = {}
        self.dir_h2d_DSNB_diff_model = {}
        self.dir_h2d_DSNB_diff_model_full ={}
        self.dir_h2d_NC_diff_model ={}
        self.dir_h2d_NC_diff_model_full ={}
        self.dir_h2d_other_bkg_to_fit = {}
        self.n_h2d_need_to_fit = 0
        self.seed = 0
        self.v_nll = []
        self.plot_v_nll = False
        self.use_sk_data = False
        self.add_fiducial_volume_2 = True
        # self.offset_PSD_fiducial_volume_2 = 3  # For fitting the FV2, we need to separate the two dataset, for convenience, we chose to offset the PSD bins.
        self.plot_comparison = False
        self.C11_cut = True
        self.separate_C11_fitting = False
        self.ratio_sig = {}
        self.h2d_sig = {}
        self.h2d_sig_full = {}
        self.h2d_sig_to_fit = {}
        self.h2d_bkg_NC = {}
        self.h2d_bkg_NC_separate = {}
        self.h2d_bkg_NC_full = {}
        self.h2d_bkg_NC_full_separate = {}
        self.h2d_bkg_NC_to_fit = {}
        self.dir_n_other_bkg = {}
        # For sklearn Data I haven't developed FV2 and C11 cut , here I turn off this two options
        if self.use_sk_data:
            self.add_fiducial_volume_2 = False
            self.C11_cut = False

    # def SetNOtherBkg(self, dir_n_other_bkg:dict):
    #     self.dir_n_other_bkg = dir_n_other_bkg
    def SetNDiffEDSNB(self, dir_n_DSNB_diff_E:dict):
        self.dir_n_DSNB_diff_E_10yr = dir_n_DSNB_diff_E
    def SetNCNumber(self, n_NC_10yr):
        self.n_NC_10yr = n_NC_10yr
    def Set1DPDFToCheckSensitivity(self):
        align_spectrum = False
        import uproot as up
        f_DSNB = up.open("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/energy_spectrum/zhangyy/DSNB_promptE_15MeV.root")
        h_DSNB, edge_DSNB = f_DSNB["signal_PDF"].to_numpy()
        self.h2d_sig_spectrum = hl.hist((np.ones(len(edge_DSNB) - 1) * 0.99, GetBinCenter(edge_DSNB)), weights=h_DSNB,bins=self.n_bins).normalize(integrate=False)
        if align_spectrum:
            self.h2d_sig["FV1"] = self.h2d_sig_spectrum
        f_DSNB.close()

        f_NC = up.open("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/energy_spectrum/zhangyy/NC_16m.root")
        h_NC, edge_NC = f_NC["NC"].to_numpy()
        self.h2d_bkg_NC_spectrum = hl.hist((np.ones(len(edge_NC)-1)*0.99, GetBinCenter(edge_NC)), weights=h_NC, bins=self.n_bins).normalize(integrate=False)
        if align_spectrum:
            self.h2d_bkg_NC["Total"] = self.h2d_bkg_NC_spectrum
        f_NC.close()

        dir_spectrum = "/afs/ihep.ac.cn/users/l/luoxj/sk_psd/energy_spectrum/"
        dir_spectrum_file = {"CC":dir_spectrum+"zhangyy/h_CC.root",  "FastN":dir_spectrum+"zhangyy/h_FastN.root",
                             "He8Li9":dir_spectrum+"zhangyy/h_Li9He8.root", "Reactor-anti-Nu":"/junofs/users/chengjie/workdir/ReactorAnti/input/MO1.0DayEres3.0207statDataIO.root"}
                             # "Reactor-anti-Nu":dir_spectrum+"chengjie/ReactorAnti.root"}
        dir_spectrum_key_in_root = {"CC":"CC",  "FastN":"FastN","He8Li9":"hSpecLiFine", "Reactor-anti-Nu":"h_NMO"}
        self.dir_h2d_other_bkg_spectrum = {}
        for key in self.dir_h2d_other_bkg.keys():
            f_other_bkg = up.open(dir_spectrum_file[key])
            h_other_bkg, edge_other_bkg = f_other_bkg[dir_spectrum_key_in_root[key]].to_numpy()
            self.dir_h2d_other_bkg_spectrum[key] = hl.hist(
                (np.ones(len(edge_other_bkg) - 1) * 0.99, GetBinCenter(edge_other_bkg)), weights=h_other_bkg,
                bins=self.n_bins).normalize(integrate=False)
            if align_spectrum:
                # and not key=="He8Li9":
                self.dir_h2d_other_bkg[key] = self.dir_h2d_other_bkg_spectrum[key]
            f_other_bkg.close()

    def LoadPrediction(self, infile:str):
        """
        :param infile: path for PSD output file
        :return:
        """
        f = np.load(infile, allow_pickle=True)
        # self.v_vertex = f["vertex"]
        self.v_equen = f["equen"]
        if self.use_sk_data:
            self.v_predict = f["predict_proba"][:, 1]
        else:
            self.v_predict = f["predict_proba"]
            self.PSD_cut = f["BDTG_cut"]
            if self.C11_cut:
                self.index_C11 = f["index_C11"]
                self.index_no_C11 = f["index_no_C11"]
        self.v_labels = f["labels"]
        # self.v_R3 = ( np.sqrt(np.sum(self.v_vertex ** 2, axis=1)) / 1000 ) **3
        self.h2d = None
        # print("check loading status")
        # print(f"length -> vertex: {len(self.v_vertex)}, equen: {len(self.v_equen)}, prediction:{len(self.v_predict)}")
        # print(f"content -> vertex: {self.v_vertex[:5]},\n equen: {self.v_equen[:5]},\n prediction:{self.v_predict[:5]}")

    def GetBkgRatio(self):
        """
        This function is for calculating the ratio of efficiency from binning strategy
        :return:
        """
        bkg_criteria = self.n_bins[0][0]
        # bkg_criteria = 0.5
        self.ratio_bkg_NC_After_PSD = {key: np.sum(self.h2d_bkg_NC_full_separate[key].values[1:,:])\
                                            /np.sum(self.h2d_bkg_NC_full_separate[key].values) for key in self.h2d_bkg_NC_separate.keys()}# the first axis' bins is [0, 0.5.....], so we use [1:, :]to get the residue
        self.ratio_bkg_NC_After_PSD["Total"] = np.sum(self.h2d_bkg_NC_full['Total'].values[1:,:])\
                                            /np.sum(self.h2d_bkg_NC_full["Total"].values)
        for key in self.h2d_sig.keys():
            self.ratio_sig[key] =  np.sum(self.h2d_sig_full[key].values[1:,:])/np.sum(self.h2d_sig_full[key].values)
        self.ratio_other_bkg = {}
        for key in self.dir_h2d_other_bkg_full.keys():
            self.ratio_other_bkg[key] = np.sum(self.dir_h2d_other_bkg_full[key].values[1:,:])/np.sum(self.dir_h2d_other_bkg_full[key].values)
        print(f"#########PSD criteria = {bkg_criteria}##############")
        for key in self.h2d_sig.keys():
            print("ratio_DSNB:\t", self.ratio_sig[key])
            print("Expected DSNB Residue:\t", self.ratio_sig[key]*self.dir_n_DSNB_diff_E_10yr["15MeV_"+key])
        print("ratio_NC:\t", self.ratio_bkg_NC_After_PSD)
        print("ratio_other_background:\t", self.ratio_other_bkg)
        for key in self.ratio_bkg_NC_After_PSD.keys():
            print(f"Expected NC({key}) Residue:\t", self.ratio_bkg_NC_After_PSD[key]*self.n_NC_10yr)
        for key in self.dir_h2d_other_bkg.keys():
            print(f"Expected {key} residue:\t",self.ratio_other_bkg[key] * self.dir_n_other_bkg[key] )
        print("#####################################################")

    def GetBestSNRatio(self):
        """

        This function is for getting the best Signal/Noise Ratio for PSD cut.
        :return: PSD cut for best S/N Ratio
        """
        v_ratio_SN = []
        v_PSD_cut = np.arange(0.2, 1, 0.01)
        predict_sig = self.v_predict[self.v_labels == 1]
        predict_bkg = self.v_predict[self.v_labels == 0]
        for PSD_cut in v_PSD_cut:
        # for PSD_cut in [0.95]:
            print(f"##############PSD cut: {PSD_cut}#################")
            # ratio_SN = Counter(predict_sig>PSD_cut)[True]/np.sqrt(Counter(predict_bkg>PSD_cut)[True]**2+Counter(predict_sig>PSD_cut)[True]**2)
            # ratio_SN = Counter(predict_sig>PSD_cut)[True]-(Counter(predict_bkg>PSD_cut)[True])
            sig_eff = Counter(predict_sig>PSD_cut)[True]/(len(predict_sig))
            bkg_eff = Counter(predict_bkg>PSD_cut)[True]/(len(predict_bkg))
            # ratio_SN = S/sqrt(S**2+N**2)
            ratio_SN = sig_eff*self.dir_n_DSNB_diff_E_10yr["15MeV_FV1"]/np.sqrt((sig_eff*self.dir_n_DSNB_diff_E_10yr["15MeV_FV1"])**2+(bkg_eff*self.n_NC_10yr)**2)
            print("S/N Ratio:\t",ratio_SN)
            print("Signal Efficiency:\t",sig_eff)
            v_ratio_SN.append(ratio_SN)
        plt.plot(v_PSD_cut, v_ratio_SN)
        plt.figure()
        plt.plot((v_PSD_cut[1:]+v_PSD_cut[:-1])/2, np.diff(v_ratio_SN))
        plt.show()
        exit()


    def LoadFV2NCAndDSNB(self, name_file:str, NC_uncertainty, n_NC_FV2):
        f = np.load(name_file, allow_pickle=True)
        # self.v_vertex = f["vertex"]
        self.v_equen_FV2 = f["equen"]
        if self.use_sk_data:
            self.v_predict_FV2 = f["predict_proba"][:, 1]
        else:
            self.v_predict_FV2 = f["predict_proba"]
            self.PSD_cut_FV2 = f["BDTG_cut"]
            if self.C11_cut:
                self.index_C11_FV2 = f["index_C11"]
                self.index_no_C11_FV2 = f["index_no_C11"]
        self.v_labels_FV2 = f["labels"]
        samples_NC_FV2 = {"prod":self.v_predict_FV2[self.v_labels_FV2==0], "equen":self.v_equen_FV2[self.v_labels_FV2==0]}
        self.key_NC_FV2 = "NC_FV2"
        self.dir_other_bkg[self.key_NC_FV2] = samples_NC_FV2
        self.dir_other_bkg_sys_uncertainty[self.key_NC_FV2] = NC_uncertainty
        self.dir_n_other_bkg[self.key_NC_FV2] = n_NC_FV2
        self.dir_is_FV2[self.key_NC_FV2] = True


    def LoadOtherBkg(self, name_file:str, key:str, key_here:str, sys_uncertainty:float, n_events_10yr:float, is_FV2=False):
        """

        :param name_file: path for other background file
        :param key: the key to get the dictionary in files
        :param key_here: the key using in this code (which is for the variable "self.dir_other_bkg")
        :param sys_uncertainty: set the system uncertainty of this background
        :return:
        """
        f = np.load(name_file, allow_pickle=True)
        samples = f[key].item()
        self.dir_other_bkg[key_here] = samples
        self.dir_other_bkg_sys_uncertainty[key_here] = sys_uncertainty
        self.dir_n_other_bkg[key_here] = n_events_10yr
        self.dir_is_FV2[key_here] = is_FV2
        print(f"Events number for PDF({key_here}):\t ", len(samples["equen"]))
        
    def GetMapFV2ToFV1Epsilon(self):
        self.map_FV2_to_epsilon = {}
        for i, key in enumerate(self.dir_h2d_other_bkg.keys()):
            if self.dir_is_FV2[key]:
                key_to_find = key.split("_")[0]
                if key_to_find=="NC":
                    self.map_FV2_to_epsilon[key] = 0
                else:
                    self.map_FV2_to_epsilon[key] = list(self.dir_h2d_other_bkg.keys()).index(key_to_find)+1
        self.n_key_FV2 = len(self.map_FV2_to_epsilon.keys())

    def LoadDSNBOtherModel(self, name_file:str, key_in_file:str, key_here:str):
        f = np.load(name_file, allow_pickle=True)
        samples = f[key_in_file].item()
        self.dir_samples_DSNB_diff_model[key_here] = samples
        print(f"Events number for PDF({key_here}):\t ", len(samples["equen"]))
    def LoadNCOtherModel(self,  name_file:str, key_in_file:str, key_here:str):
        f = np.load(name_file, allow_pickle=True)
        samples = f[key_in_file].item()
        self.dir_samples_NC_diff_model[key_here] = samples
        print(f"Events number for PDF({key_here}):\t ", len(samples["equen"]))

    def Plot2DPDF(self):
        for i, key in enumerate(self.h2d_sig.keys()):
            fig2, locals()[f"ax2_{i}"] = plt.subplots()
            hl.plot2d(locals()[f"ax2_{i}"], self.h2d_sig[key], log=True, cbar=True, clabel="counts per bin")
            SetTitle(f"PDF(Signal_{key})", locals()[f"ax2_{i}"])
        keys_to_plot_NC = self.h2d_bkg_NC_separate.keys() if self.C11_cut else self.h2d_bkg_NC.keys()
        for i,key in enumerate(keys_to_plot_NC):
            fig2_extra, locals()[f"ax3_{i}"] = plt.subplots()
            if self.C11_cut:
                hl.plot2d(locals()[f"ax3_{i}"], self.h2d_bkg_NC_separate[key], log=True, cbar=True, clabel="counts per bin")
            else:
                hl.plot2d(locals()[f"ax3_{i}"], self.h2d_bkg_NC[key], log=True, cbar=True, clabel="counts per bin")
            if key=="Total":
                SetTitle(f"PDF(NC_FV1)", locals()[f"ax3_{i}"])
            else:
                SetTitle(f"PDF (NC [{key}])", locals()[f"ax3_{i}"])

        n_other_bkg =len(self.dir_h2d_other_bkg_full.keys())
        for i, key in enumerate(self.dir_other_bkg.keys()):
            fig_other_bkg, ax_other_bkg = plt.subplots()
            hl.plot2d(ax_other_bkg, self.dir_h2d_other_bkg[key],log=True, cbar=True, clabel="counts per bin")
            if self.dir_is_FV2[key]:
                SetTitle(f"PDF({key})", ax_other_bkg)
            else:
                SetTitle(f"PDF({key}_FV1)", ax_other_bkg)
        if align_sensitivity and not fit_2d:
            for i, key in enumerate(self.dir_samples_DSNB_diff_model.keys()):
                fig_diff_DSNB, ax_diff_DSNB = plt.subplots()
                hl.plot2d(ax_diff_DSNB, self.dir_h2d_DSNB_diff_model[key], log=True, cbar=True, clabel="counts per bin")
                SetTitle(f"PDF({key} DSNB model)", ax_diff_DSNB)
            for i, key in enumerate(self.dir_samples_NC_diff_model.keys()):
                fig_diff_NC, ax_diff_NC = plt.subplots()
                hl.plot2d(ax_diff_NC, self.dir_h2d_NC_diff_model[key], log=True, cbar=True, clabel="counts per bin")
                SetTitle(f"PDF({key} NC model)", ax_diff_NC)

        plt.show()

    def SetN_Bins(self):
        # binning strategy for fitting
        low_edge_E = 12
        if self.use_sk_data:
            down_PSD_limit = 0.
            if fit_2d:
                n_bins = [np.array([  0.5,0.9, 0.95, 0.985,  1.001]), np.linspace(low_edge_E, 30, 11)]
                # n_bins = [np.array([  0.5, 0.95, 0.985,  1.001]), np.linspace(10, 30, 11)]
            else:
                n_bins = [np.array([0.95, 1.001]), np.linspace(low_edge_E, 30, 11)]
                #n_bins = [np.array([ 0.94, 1.001]), np.linspace(10, 30, 21)]
        else:
            down_PSD_limit = -1.
            if fit_2d:
                # n_bins = [np.array([  0.5,0.9, 0.95, 0.985,  1.001]), np.linspace(10, 30, 11)]
                # n_bins = [np.array([ 0., 0.7, 0.9, 0.97, 1.001]), np.linspace(10, 30, 11)]
                # n_bins = [np.array([ 0., 0.7, 0.9, 0.97, 1.001]), np.linspace(low_edge_E, 30, 11)]
                n_bins = [np.array([ 0., 0.7, 0.9, 0.97, 1.001]), np.arange(low_edge_E, 32, 2)]
                # n_bins = [np.array([0.5, 0.95, 0.985, 1.001]), np.linspace(10, 30, 11)]
            else:
                n_bins = [np.array([self.PSD_cut, 1.001]), np.arange(low_edge_E, 32, 2)]
                # n_bins = [np.array([self.PSD_cut, 1.001]), np.linspace(low_edge_E, 30, 11)]

        # set bins for the whole distribution for sampling
        if  n_bins[0][0]!=down_PSD_limit:
            self.n_bins_full = [np.concatenate((np.array([down_PSD_limit]), n_bins[0])), n_bins[1]]
        else:
            self.n_bins_full = n_bins
        self.index_to_separate_volume = len(n_bins)
        self.index_to_separate_volume_full = len(self.n_bins_full)
        # if self.add_fiducial_volume_2:
        #     n_bins[0] = np.concatenate((n_bins[0], n_bins[0] + self.offset_PSD_fiducial_volume_2))
        #     self.n_bins_full[0] = np.concatenate((self.n_bins_full[0], self.n_bins_full[0]+self.offset_PSD_fiducial_volume_2))

        self.n_bins = n_bins
        print("n_bins:\t", self.n_bins)

    def Get2DPDFHist(self, fit_2d):
        """
        Get PDF for fitting
        :param fit_2d: boolean for whether using the 2d fitting
        :return:
        """
        plot_2d_pdf = False

        # set the bins and PSD output for the class
        predict_1 = self.v_predict
        self.predict_1 = predict_1
        indices_bkg_NC = (self.v_labels==0)
        if self.C11_cut:
            indices_bkg_NC_C11 = self.index_C11[indices_bkg_NC]
            indices_bkg_NC_no_C11 = self.index_no_C11[indices_bkg_NC]
            ratio_C11 = Counter(indices_bkg_NC_C11)[1]/len(indices_bkg_NC_C11)
            ratio_no_C11 = Counter(indices_bkg_NC_no_C11)[1]/len(indices_bkg_NC_no_C11)
            self.dir_ratio_C11 = {"w/ C11":ratio_C11, "w/o C11":ratio_no_C11}
            self.dir_eff_tccut = {"w/ C11":0.255, "w/o C11":0.936}
            for key in self.dir_n_DSNB_diff_E_10yr.keys():
                if "FV1" in key:
                    self.dir_n_DSNB_diff_E_10yr[key] = self.dir_n_DSNB_diff_E_10yr[key]*self.dir_eff_tccut["w/o C11"]
            for key in self.dir_n_other_bkg.keys():
                if not self.dir_is_FV2[key]:
                    self.dir_n_other_bkg[key] *= self.dir_eff_tccut["w/o C11"]
            print("################ C11 Cut ############################")
            print("Ratio of C11 in NC bkg:\t", self.dir_ratio_C11)
            print("Eff. of tccut:\t", self.dir_eff_tccut)
            print("#####################################################")
        else:
            self.dir_ratio_C11 = {"Total":1}

        indices_sig = (self.v_labels==1)
        if self.add_fiducial_volume_2:
            indices_sig_FV2 = (self.v_labels_FV2==1)
        equen = self.v_equen

        # Getting PDF for signal and background( x axis is PSD output , y axis is Equen)
        if without_normalize:
            self.h2d_sig["FV1"] = hl.hist((predict_1[indices_sig], equen[indices_sig]), bins=self.n_bins)
            self.h2d_sig_full["FV1"] = hl.hist((predict_1[indices_sig], equen[indices_sig]), bins=self.n_bins_full)
            if flh.add_fiducial_volume_2:
                self.h2d_sig["FV2"] = hl.hist((self.v_predict_FV2[indices_sig_FV2], self.v_equen_FV2[indices_sig_FV2]), bins=self.n_bins)
                self.h2d_sig_full["FV2"] = hl.hist((self.v_predict_FV2[indices_sig_FV2], self.v_equen_FV2[indices_sig_FV2]), bins=self.n_bins_full)

            if not self.C11_cut:
                self.h2d_bkg_NC["Total"] = hl.hist((predict_1[indices_bkg_NC], equen[indices_bkg_NC]), bins=self.n_bins)
                self.h2d_bkg_NC_full["Total"] = hl.hist((predict_1[indices_bkg_NC], equen[indices_bkg_NC]), bins=self.n_bins_full)
            elif self.C11_cut and self.separate_C11_fitting:
                self.h2d_bkg_NC["w/ C11"] = hl.hist((predict_1[self.index_C11], equen[self.index_C11]), bins=self.n_bins)
                self.h2d_bkg_NC["w/o C11"]= hl.hist((predict_1[self.index_no_C11], equen[self.index_no_C11]), bins=self.n_bins)
                self.h2d_bkg_NC_full["w/ C11"] = hl.hist((predict_1[self.index_C11], equen[self.index_C11]), bins=self.n_bins_full)
                self.h2d_bkg_NC_full["w/o C11"] = hl.hist((predict_1[self.index_no_C11], equen[self.index_no_C11]), bins=self.n_bins_full)
                for key in self.h2d_bkg_NC.keys():
                    self.h2d_bkg_NC_separate[key] = self.h2d_bkg_NC[key]
                    self.h2d_bkg_NC_full_separate[key] = self.h2d_bkg_NC_full[key]
            elif self.C11_cut and not self.separate_C11_fitting:
                self.h2d_bkg_NC_separate["w/ C11"] = hl.hist((predict_1[self.index_C11], equen[self.index_C11]), bins=self.n_bins)
                self.h2d_bkg_NC_separate["w/o C11"]= hl.hist((predict_1[self.index_no_C11], equen[self.index_no_C11]), bins=self.n_bins)
                self.h2d_bkg_NC_full_separate["w/ C11"] = hl.hist((predict_1[self.index_C11], equen[self.index_C11]), bins=self.n_bins_full)
                self.h2d_bkg_NC_full_separate["w/o C11"] = hl.hist((predict_1[self.index_no_C11], equen[self.index_no_C11]), bins=self.n_bins_full)
                self.h2d_bkg_NC["Total"] = self.h2d_bkg_NC_separate["w/ C11"] + self.h2d_bkg_NC_separate["w/o C11"]
                self.h2d_bkg_NC_full["Total"] = self.h2d_bkg_NC_full_separate["w/ C11"] + self.h2d_bkg_NC_full_separate["w/o C11"]


            for key in self.dir_other_bkg.keys():
                self.dir_h2d_other_bkg[key] = hl.hist((self.dir_other_bkg[key]["prod"], self.dir_other_bkg[key]["equen"]), bins=self.n_bins)
                self.dir_h2d_other_bkg_full[key] = hl.hist((self.dir_other_bkg[key]["prod"], self.dir_other_bkg[key]["equen"]), bins=self.n_bins_full)
            for key in self.dir_samples_DSNB_diff_model.keys():
                self.dir_h2d_DSNB_diff_model[key] = hl.hist((self.dir_samples_DSNB_diff_model[key]["prod"], self.dir_samples_DSNB_diff_model[key]["equen"]), bins=self.n_bins)
                self.dir_h2d_DSNB_diff_model_full[key] = hl.hist((self.dir_samples_DSNB_diff_model[key]["prod"], self.dir_samples_DSNB_diff_model[key]["equen"]), bins=self.n_bins_full)
            for key in self.dir_samples_NC_diff_model.keys():
                self.dir_h2d_NC_diff_model[key] = hl.hist(self.dir_samples_NC_diff_model["key"]["prod"], self.dir_samples_NC_diff_model[key]["equen"], bins=self.n_bins)
                self.dir_h2d_NC_diff_model_full[key] = hl.hist(self.dir_samples_NC_diff_model["key"]["prod"], self.dir_samples_NC_diff_model[key]["equen"], bins=self.n_bins_full)
            # print("sig:\t", np.sum(self.h2d_sig.values, axis=1))
            # print("NC:\t",  np.sum(self.h2d_bkg_NC.values, axis=1))
            # exit()
        else:
            self.h2d_sig["FV1"] = hl.hist((predict_1[indices_sig], equen[indices_sig]), bins=self.n_bins).normalize(integrate=False)
            self.h2d_sig_full["FV1"] = hl.hist((predict_1[indices_sig], equen[indices_sig]), bins=self.n_bins_full).normalize(integrate=False)
            if flh.add_fiducial_volume_2:
                self.h2d_sig["FV2"] = hl.hist((self.v_predict_FV2[indices_sig_FV2], self.v_equen_FV2[indices_sig_FV2]), bins=self.n_bins).normalize(integrate=False)
                self.h2d_sig_full["FV2"] = hl.hist((self.v_predict_FV2[indices_sig_FV2], self.v_equen_FV2[indices_sig_FV2]), bins=self.n_bins_full).normalize(integrate=False)
            if not self.C11_cut:
                self.h2d_bkg_NC["Total"] = hl.hist((predict_1[indices_bkg_NC], equen[indices_bkg_NC]), bins=self.n_bins).normalize(integrate=False)
                self.h2d_bkg_NC_full["Total"] = hl.hist((predict_1[indices_bkg_NC], equen[indices_bkg_NC]), bins=self.n_bins_full).normalize(integrate=False)
                for key in self.h2d_bkg_NC.keys():
                    self.h2d_bkg_NC_separate[key] = self.h2d_bkg_NC[key]
                    self.h2d_bkg_NC_full_separate[key] = self.h2d_bkg_NC_full[key]
            elif self.C11_cut and self.separate_C11_fitting:
                self.h2d_bkg_NC["w/ C11"] = hl.hist((predict_1[self.index_C11], equen[self.index_C11]), bins=self.n_bins).normalize(integrate=False)
                self.h2d_bkg_NC["w/o C11"]= hl.hist((predict_1[self.index_no_C11], equen[self.index_no_C11]), bins=self.n_bins).normalize(integrate=False)
                self.h2d_bkg_NC_full["w/ C11"] = hl.hist((predict_1[self.index_C11], equen[self.index_C11]), bins=self.n_bins_full).normalize(integrate=False)
                self.h2d_bkg_NC_full["w/o C11"] = hl.hist((predict_1[self.index_no_C11], equen[self.index_no_C11]), bins=self.n_bins_full).normalize(integrate=False)
                for key in self.h2d_bkg_NC.keys():
                    self.h2d_bkg_NC_separate[key] = self.h2d_bkg_NC[key]
                    self.h2d_bkg_NC_full_separate[key] = self.h2d_bkg_NC_full[key]
            elif self.C11_cut and not self.separate_C11_fitting:
                check_pdf_adding = True
                if not check_pdf_adding:
                    self.h2d_bkg_NC_separate["w/ C11"] = hl.hist((predict_1[self.index_C11], equen[self.index_C11]), bins=self.n_bins).normalize(integrate=False)
                    self.h2d_bkg_NC_separate["w/o C11"]= hl.hist((predict_1[self.index_no_C11], equen[self.index_no_C11]), bins=self.n_bins).normalize(integrate=False)
                    self.h2d_bkg_NC_full_separate["w/ C11"] = hl.hist((predict_1[self.index_C11], equen[self.index_C11]), bins=self.n_bins_full).normalize(integrate=False)
                    self.h2d_bkg_NC_full_separate["w/o C11"] = hl.hist((predict_1[self.index_no_C11], equen[self.index_no_C11]), bins=self.n_bins_full).normalize(integrate=False)
                    self.h2d_bkg_NC["Total"] = hl.hist((predict_1[indices_bkg_NC], equen[indices_bkg_NC]), bins=self.n_bins).normalize(integrate=False)
                    self.h2d_bkg_NC_full["Total"] = hl.hist((predict_1[indices_bkg_NC], equen[indices_bkg_NC]), bins=self.n_bins_full).normalize(integrate=False)
                else:
                    self.h2d_bkg_NC_separate["w/ C11"] = hl.hist((predict_1[self.index_C11], equen[self.index_C11]), bins=self.n_bins)
                    self.h2d_bkg_NC_separate["w/o C11"]= hl.hist((predict_1[self.index_no_C11], equen[self.index_no_C11]), bins=self.n_bins)
                    self.h2d_bkg_NC_full_separate["w/ C11"] = hl.hist((predict_1[self.index_C11], equen[self.index_C11]), bins=self.n_bins_full)
                    self.h2d_bkg_NC_full_separate["w/o C11"] = hl.hist((predict_1[self.index_no_C11], equen[self.index_no_C11]), bins=self.n_bins_full)
                    self.h2d_bkg_NC["Total"] = (self.h2d_bkg_NC_separate["w/ C11"]*self.dir_eff_tccut["w/ C11"]
                                              + self.h2d_bkg_NC_separate["w/o C11"]*self.dir_eff_tccut["w/o C11"])\
                                                /(self.dir_eff_tccut["w/ C11"]+self.dir_eff_tccut["w/o C11"])
                    self.h2d_bkg_NC_full["Total"] = (self.h2d_bkg_NC_full_separate["w/ C11"]*self.dir_eff_tccut["w/ C11"] \
                                                   + self.h2d_bkg_NC_full_separate["w/o C11"]*self.dir_eff_tccut["w/o C11"])\
                                                    / ( self.dir_eff_tccut["w/ C11"] +  self.dir_eff_tccut["w/o C11"])
                    for key in self.h2d_bkg_NC.keys():
                        self.h2d_bkg_NC[key] = self.h2d_bkg_NC[key].normalize(integrate=False)
                        self.h2d_bkg_NC_full[key] = self.h2d_bkg_NC_full[key].normalize(integrate=False)
                    for key in self.h2d_bkg_NC_separate.keys():
                        self.h2d_bkg_NC_separate[key] = self.h2d_bkg_NC_separate[key].normalize(integrate=False)
                        self.h2d_bkg_NC_full_separate[key] = self.h2d_bkg_NC_full_separate[key].normalize(integrate=False)

            for key in self.dir_other_bkg.keys():
                self.dir_h2d_other_bkg[key] = hl.hist((self.dir_other_bkg[key]["prod"], self.dir_other_bkg[key]["equen"]), bins=self.n_bins).normalize(integrate=False)
                self.dir_h2d_other_bkg_full[key] = hl.hist((self.dir_other_bkg[key]["prod"], self.dir_other_bkg[key]["equen"]), bins=self.n_bins_full).normalize(integrate=False)
            for key in self.dir_samples_DSNB_diff_model.keys():
                self.dir_h2d_DSNB_diff_model[key] = hl.hist((self.dir_samples_DSNB_diff_model[key]["prod"], self.dir_samples_DSNB_diff_model[key]["equen"]), bins=self.n_bins).normalize(integrate=False)
                self.dir_h2d_DSNB_diff_model_full[key] = hl.hist((self.dir_samples_DSNB_diff_model[key]["prod"], self.dir_samples_DSNB_diff_model[key]["equen"]), bins=self.n_bins_full).normalize(integrate=False)

            for key in self.dir_samples_NC_diff_model.keys():
                self.dir_h2d_NC_diff_model[key] = hl.hist((self.dir_samples_NC_diff_model[key]["prod"],
                                                          self.dir_samples_NC_diff_model[key]["equen"]), bins=self.n_bins).normalize(integrate=False)
                self.dir_h2d_NC_diff_model_full[key] = hl.hist((self.dir_samples_NC_diff_model[key]["prod"],
                                                               self.dir_samples_NC_diff_model[key]["equen"]),
                                                               bins=self.n_bins_full).normalize(integrate=False)
        self.n_h2d_sig = len(self.h2d_sig.keys())
        self.n_h2d_need_to_fit = len(self.dir_h2d_other_bkg.keys()) + self.n_h2d_sig + len(self.h2d_bkg_NC.keys())

        # Use other DSNB model as pdf
        if use_additive_DSNB_And_NC and not fit_2d and not self.C11_cut:
            plot_compare_diff_NC_model = False
            self.h2d_sig["FV1"] = self.dir_h2d_DSNB_diff_model["15MeV"]
            self.h2d_sig_full["FV1"] = self.dir_h2d_DSNB_diff_model_full["15MeV"]
            if plot_compare_diff_NC_model:
                fig_NC_compare, ax_NC_compare = plt.subplots()
                hl.plot1d(ax_NC_compare, self.h2d_bkg_NC["Total"].project([1]), label="Genie Model")
                hl.plot1d(ax_NC_compare, self.dir_h2d_NC_diff_model["Average_Model"].project([1]), label="Average Model")
                plt.legend()
                plt.xlabel("$E_{quen}$")
                plt.show()
            self.h2d_bkg_NC["Total"] = self.dir_h2d_NC_diff_model["Average_Model"]
            self.h2d_bkg_NC_full["Total"] = self.dir_h2d_NC_diff_model_full["Average_Model"]

        self.GetBkgRatio()

        if save_2d_PDF:
            import ROOT
            from array import  array
            dir_to_save = "./pdf_root/"
            if not os.path.isdir(dir_to_save):
                os.mkdir(dir_to_save)
            f_pdf = ROOT.TFile(f"{dir_to_save}pdf_{label_fit_method}.root", "recreate")
            h2d_sig_to_save = ROOT.TH2D("h_DSNB", "h_DSNB", len(self.n_bins[0])-1, array("d", self.n_bins[0]),
                                        len(self.n_bins[1])-1, array("d", self.n_bins[1]))
            h2d_NC_to_save = ROOT.TH2D("h_NC", "h_NC", len(self.n_bins[0])-1, array("d", self.n_bins[0]),
                                        len(self.n_bins[1])-1, array("d", self.n_bins[1]))
            for i in range(len(self.n_bins[0])-1):
                for j in range(len(self.n_bins[1])-1):
                    h2d_sig_to_save.SetBinContent(i+1, j+1, self.h2d_sig["FV1"].values[i][j])
                    h2d_NC_to_save.SetBinContent(i+1, j+1, self.h2d_bkg_NC["Total"].values[i][j])
            for key in self.dir_h2d_other_bkg.keys():
                h2d_other_bkg_to_save = ROOT.TH2D(f"h_{key}", f"h_{key}", len(self.n_bins[0]) - 1, array("d", self.n_bins[0]),
                                           len(self.n_bins[1]) - 1, array("d", self.n_bins[1]))
                for i in range(len(self.n_bins[0])-1):
                    for j in range(len(self.n_bins[1])-1):
                        h2d_other_bkg_to_save.SetBinContent(i+1, j+1, self.dir_h2d_other_bkg[key].values[i][j])
                h2d_other_bkg_to_save.Write()
            f_pdf.cd()
            h2d_sig_to_save.Write()
            h2d_NC_to_save.Write()
            f_pdf.Close()
            exit()

        # Plot all PDFs using in this fitting
        if plot_2d_pdf:
            self.Plot2DPDF()


    def PlotPDFProfile(self):
        plot_errorbars = False
        plot_seperate_comparison = False
        fig_profile_equen, ax_profile_equen = plt.subplots()
        if plot_seperate_comparison and self.plot_comparison:
            fig_profile_equen_DSNB, ax_profile_equen_DSNB = plt.subplots()
            fig_profile_equen_NC,   ax_profile_equen_NC = plt.subplots()
        if not without_normalize:
            for key in self.h2d_sig.keys():
                hl.plot1d(ax_profile_equen, self.h2d_sig[key].project([1]), label=f"DSNB({key})",errorbars=plot_errorbars)
            for key in self.h2d_bkg_NC.keys():
                hl.plot1d(ax_profile_equen, self.h2d_bkg_NC[key].project([1]), label=f"atm-NC({key})",errorbars=plot_errorbars)
            if not fit_2d and self.plot_comparison:
                hl.plot1d(ax_profile_equen, self.h2d_sig_spectrum.project([1]), label="DSNB_spectrum", errorbars=plot_errorbars, ls="--")
                hl.plot1d(ax_profile_equen, self.h2d_bkg_NC_spectrum.project([1]), label="atm-NC_spectrum", errorbars=plot_errorbars, ls="--")
                if plot_seperate_comparison:
                    for key in self.h2d_sig.keys():
                        hl.plot1d(ax_profile_equen_DSNB, self.h2d_sig[key].project([1]), label=f"DSNB({key})", errorbars=plot_errorbars)
                    for key in self.h2d_bkg_NC.keys():
                        hl.plot1d(ax_profile_equen_NC, self.h2d_bkg_NC[key].project([1]), label=f"atm-NC({key})", errorbars=plot_errorbars)
                    hl.plot1d(ax_profile_equen_DSNB, self.h2d_sig_spectrum.project([1]), label="DSNB_spectrum", errorbars=plot_errorbars,
                              ls="--")
                    hl.plot1d(ax_profile_equen_NC, self.h2d_bkg_NC_spectrum.project([1]), label="atm-NC_spectrum",
                              errorbars=plot_errorbars, ls="--")
                    ax_profile_equen_DSNB.set_xlabel("$E_{quen}$")
                    ax_profile_equen_NC.set_xlabel("$E_{quen}$")
                    ax_profile_equen_DSNB.set_title("DSNB Projection of $E_{quen}$")
                    ax_profile_equen_DSNB.legend()
                    ax_profile_equen_NC.set_title("Atm-NC Projection of $E_{quen}$")
                    ax_profile_equen_NC.legend()
                    fig_profile_equen_DSNB.savefig(dir_save_fig+"DSNB_equen_spectrum.png")
                    fig_profile_equen_NC.savefig(dir_save_fig+"NC_equen_spectrum.png")


        for key in self.dir_h2d_other_bkg.keys():
            hl.plot1d(ax_profile_equen, self.dir_h2d_other_bkg[key].project([1]), label=key,errorbars=plot_errorbars)
            if not fit_2d and self.plot_comparison:
                hl.plot1d(ax_profile_equen, self.dir_h2d_other_bkg_spectrum[key].project([1]), label=key+"_spectrum", errorbars=plot_errorbars, ls="--")
                if plot_seperate_comparison:
                    locals()[f"fig_profile_equen_{key}"], locals()[f"ax_profile_equen_{key}"] = plt.subplots()
                    hl.plot1d(locals()[f"ax_profile_equen_{key}"], self.dir_h2d_other_bkg[key].project([1]), label=key,errorbars=plot_errorbars)
                    hl.plot1d(locals()[f"ax_profile_equen_{key}"], self.dir_h2d_other_bkg_spectrum[key].project([1]), label=key+"_spectrum", errorbars=plot_errorbars, ls="--")
                    locals()[f"ax_profile_equen_{key}"].set_xlabel("$E_{quen}$ ")
                    locals()[f"ax_profile_equen_{key}"].set_title(key+" Projection of $E_{quen}$")
                    locals()[f"ax_profile_equen_{key}"].legend()
                    locals()[f"fig_profile_equen_{key}"].savefig(dir_save_fig +f"{key}_equen_spectrum.png")

        ax_profile_equen.set_xlabel("$E_{quen}$ ")
        ax_profile_equen.set_title("Projection of $E_{quen}$")
        ax_profile_equen.legend()

        fig_profile_PSD_full, ax_profile_PSD_full = plt.subplots()
        if not without_normalize:
            for key in self.h2d_sig.keys():
                hl.plot1d(ax_profile_PSD_full, self.h2d_sig_full[key].project([0]), label=f"DSNB_{key}",errorbars=plot_errorbars)
            hl.plot1d(ax_profile_PSD_full, self.h2d_bkg_NC_full["Total"].project([0]), label="atm-NC",errorbars=plot_errorbars)
        for key in self.dir_h2d_other_bkg_full.keys():
            hl.plot1d(ax_profile_PSD_full, self.dir_h2d_other_bkg_full[key].project([0]), label=key,errorbars=plot_errorbars)
        ax_profile_PSD_full.set_xlabel("PSD Output")
        plt.title("Projection of PSD Output(Full PDF)")
        plt.legend()

        fig_profile_PSD, ax_profile_PSD = plt.subplots()
        if not without_normalize:
            for key in self.h2d_sig.keys():
                hl.plot1d(ax_profile_PSD, self.h2d_sig[key].project([0]), label=f"DSNB_{key}",errorbars=plot_errorbars)
            hl.plot1d(ax_profile_PSD, self.h2d_bkg_NC["Total"].project([0]), label="atm-NC",errorbars=plot_errorbars)
        for key in self.dir_h2d_other_bkg_full.keys():
            hl.plot1d(ax_profile_PSD, self.dir_h2d_other_bkg[key].project([0]), label=key,errorbars=plot_errorbars)
        ax_profile_PSD.set_xlabel("PSD Output")
        plt.title("Projection of PSD Output")
        plt.legend()
        plt.show()

    def PlotHistToFit(self):
        """
        Plot histograms of samples to fit (checking the fitting samples)

        :return:
        """
        fig1, ax1 = plt.subplots()
        hl.plot2d(ax1, self.h2d_to_fit, cbar=True, clabel="counts per bin")
        SetTitle("Total Hist To Fit")

        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16,6))
        hl.plot2d(ax2, self.h2d_sig_to_fit["FV1"], cbar=True, clabel="counts per bin")
        SetTitle("Signal", ax2)
        hl.plot2d(ax3, self.h2d_bkg_NC_to_fit["Total"], cbar=True, clabel="counts per bin")
        SetTitle("NC", ax3)

        n_other_bkg =len(self.dir_h2d_other_bkg_to_fit.keys())
        for i, key in enumerate(self.dir_other_bkg.keys()):
            fig_other_bkg, ax_other_bkg = plt.subplots()
            hl.plot2d(ax_other_bkg, self.dir_h2d_other_bkg_to_fit[key], cbar=True, clabel="counts per bin")
            SetTitle(f"{key}", ax_other_bkg)
        plt.show()

    def GetAsimovDatasetToFit(self,time_scale_year:float ,key_E_DSNB:str="15MeV", print_fit_result:bool=False):
        self.sig_eff_81 = True
        n_input = 0
        save_Data_to_ROOT = False
        self.print_fit_result = print_fit_result
        check_AsimovDataset_to_fit = False
        ratio_time = time_scale_year/10

        if align_sensitivity and not fit_2d and self.sig_eff_81:
            self.input_sig_n_align_sensitivity = 17.4
            self.input_bkg_NC_align_sensitivity = 5
            # self.input_sig_n_align_sensitivity = 17.90
            # self.input_bkg_NC_align_sensitivity = 4.33
            for key in self.h2d_sig.keys():
                self.h2d_sig_to_fit[key] = self.input_sig_n_align_sensitivity*ratio_time*self.h2d_sig[key]
            self.h2d_bkg_NC_to_fit["Total"] = self.input_bkg_NC_align_sensitivity*self.h2d_bkg_NC["Total"]*ratio_time
        else:
            for key in self.h2d_sig.keys():
                self.h2d_sig_to_fit[key] = self.ratio_sig[key]*self.dir_n_DSNB_diff_E_10yr[key_E_DSNB+"_"+key]*ratio_time*self.h2d_sig[key]
            for key in self.h2d_bkg_NC_separate.keys():
                if self.C11_cut:
                    self.h2d_bkg_NC_to_fit[key] = self.ratio_bkg_NC_After_PSD[key]*self.h2d_bkg_NC_separate[key]*self.n_NC_10yr*ratio_time * self.dir_ratio_C11[key]*self.dir_eff_tccut[key]
                else:
                    self.h2d_bkg_NC_to_fit[key] = self.ratio_bkg_NC_After_PSD[key]*self.h2d_bkg_NC_separate[key]*self.n_NC_10yr*ratio_time * self.dir_ratio_C11[key]
            # for key in self.h2d_bkg_NC.keys():
            #     self.h2d_bkg_NC_to_fit[key] = self.ratio_bkg_NC_After_PSD[key]*self.h2d_bkg_NC[key]*self.n_NC_10yr*ratio_time

        for key in self.h2d_sig.keys():
            n_input += (self.ratio_sig[key]*self.dir_n_DSNB_diff_E_10yr[key_E_DSNB+"_"+key])*ratio_time
        self.h2d_to_fit =  self.h2d_sig_to_fit["FV1"]
        if self.add_fiducial_volume_2:
            self.h2d_to_fit_FV2 = self.h2d_sig_to_fit["FV2"]
        for key in self.h2d_bkg_NC_to_fit.keys():
            self.h2d_to_fit += self.h2d_bkg_NC_to_fit[key]
            if self.C11_cut:
                n_input += (self.ratio_bkg_NC_After_PSD[key]*self.n_NC_10yr)*ratio_time*self.dir_eff_tccut[key]
            else:
                n_input += (self.ratio_bkg_NC_After_PSD[key] * self.n_NC_10yr) * ratio_time
        # print("Signal PDF:\t",self.h2d_sig_to_fit.values)

        if align_sensitivity and not fit_2d and self.sig_eff_81:
            self.dir_n_other_bkg_1d_align_sensitivity ={"CC":2.0,  "FastN":0.04, "He8Li9":0.06, "Reactor-anti-Nu":2.8}
            # self.dir_n_other_bkg_1d_align_sensitivity["CC"] = 2.13
            # self.dir_n_other_bkg_1d_align_sensitivity["FastN"] = 0.075
            # self.dir_n_other_bkg_1d_align_sensitivity["He8Li9"] = 0.069
            # self.dir_n_other_bkg_1d_align_sensitivity["Reactor-anti-Nu"] = 2.90

        for key in self.dir_h2d_other_bkg.keys():
            if align_sensitivity and not fit_2d and self.sig_eff_81:
                self.dir_h2d_other_bkg_to_fit[key] = self.dir_h2d_other_bkg[key] * self.dir_n_other_bkg_1d_align_sensitivity[key]* ratio_time
            else:
                self.dir_h2d_other_bkg_to_fit[key] =  self.dir_h2d_other_bkg[key]*self.ratio_other_bkg[key]*ratio_time*self.dir_n_other_bkg[key]
            if self.dir_is_FV2[key]:
                self.h2d_to_fit_FV2 += self.dir_h2d_other_bkg_to_fit[key]
            else:
                self.h2d_to_fit += self.dir_h2d_other_bkg_to_fit[key]
            n_input += self.ratio_other_bkg[key]*ratio_time*self.dir_n_other_bkg[key]
            print(f"input {key}:\t",self.ratio_other_bkg[key]*ratio_time*self.dir_n_other_bkg[key] )

        for key in self.h2d_sig.keys():
            print(f"input DSNB_({key})(PSD):\t",(self.ratio_sig[key]*self.dir_n_DSNB_diff_E_10yr[key_E_DSNB+"_"+key])*ratio_time )
        for key in self.h2d_bkg_NC_separate.keys():
            print(f"input NC({key}) (PSD):\t",self.ratio_bkg_NC_After_PSD[key]*self.n_NC_10yr*ratio_time*self.dir_ratio_C11[key])
        print("Input number of events:\t", n_input)
        print("Hist to be fit Sum:\t", np.sum(self.h2d_to_fit.values))
        # print("AsimovDataset:\t", ','.join(map(str, np.concatenate(self.h2d_to_fit.values))))
        if save_Data_to_ROOT:
            import ROOT
            n_bins_to_save = 20
            f_to_save = ROOT.TFile(f"./AsimovDataset_DSNB/AsimovDataset_DSNB_{label_fit_method}.root", "recreate")
            print(self.n_bins[1][0],self.n_bins[1][-1])
            h_data = ROOT.TH1D("h_withoutC11cut", "h_withoutC11cut", n_bins_to_save,self.n_bins[1][0], self.n_bins[1][-1])
            print(self.h2d_to_fit.values)
            for i in range(n_bins_to_save):
                h_data.SetBinContent(i+1, self.h2d_to_fit.values[0][i])
            f_to_save.cd()
            h_data.Write()
            f_to_save.Close()
            exit()

        if check_AsimovDataset_to_fit:
            fig_profile_PSD_Asimov , ax_profile_PSD_Asimov  = plt.subplots()
            fig_profile_Equen_Asimov , ax_profile_Equen_Asimov = plt.subplots()
            v_ax = [ax_profile_PSD_Asimov, ax_profile_Equen_Asimov]
            v_xlable = ["PSD Output", "$E_{quen}$"]
            v_fig = [fig_profile_PSD_Asimov, fig_profile_Equen_Asimov]
            plot_errorbars = False
            for i_ax in [0,1]:
                plt.figure(v_fig[i_ax].number)
                hl.plot1d(v_ax[i_ax] , self.h2d_to_fit.project([i_ax]), errorbars=plot_errorbars, label="Total Input",color="black")
                for key in self.h2d_bkg_NC_separate.keys():
                    hl.plot1d(v_ax[i_ax] , self.h2d_bkg_NC_to_fit[key].project([i_ax]), errorbars=plot_errorbars, label=f"NC({key}) Input",ls="--")
                for key in self.h2d_sig.keys():
                    hl.plot1d(v_ax[i_ax] , self.h2d_sig_to_fit[key].project([i_ax]), errorbars=plot_errorbars, label=f"DSNB Input({key})",ls="--")
                for key in self.dir_h2d_other_bkg_to_fit.keys():
                    hl.plot1d(v_ax[i_ax], self.dir_h2d_other_bkg_to_fit[key].project([i_ax]), errorbars=plot_errorbars, label=f"{key} Input",ls="--")
                v_ax[i_ax].set_xlabel(v_xlable[i_ax])
                plt.legend()
            plt.show()

    def GetPositiveGausSysUncertainty(self, sigma:float):
        # Adding systematic uncertainty for toy sampling
        sys_uncertainty_factor = random.gauss(1, sigma)
        while sys_uncertainty_factor<0:
            sys_uncertainty_factor = random.gauss(1, sigma)
        return sys_uncertainty_factor

    def MergeSamples(self, sample1:dict, sample2:dict):
        sample_return = {}
        for key in sample1.keys():
            sample_return[key] = np.concatenate((sample1[key], sample2[key]))
        return sample_return

    def GetHistToFit(self, n_to_fit_sig:int, n_to_fit_bkg_NC:int, seed:int, print_fit_result:bool):
        """

        :param n_to_fit_sig:  the mean value of signal events number
        :param n_to_fit_bkg_NC:  the mean value of NC background number
        :param seed: the random seed for sampling
        :param print_fit_result: boolean for whether print the fitting result
        :return:
        """
        self.print_fit_result = print_fit_result
        # plot_2d_to_fit = True
        # plot_2d_to_fit_full_bins = True
        self.seed = seed
        plot_2d_to_fit = False
        plot_2d_to_fit_full_bins = False

        n_sig_samples = np.random.poisson(n_to_fit_sig)

        sig_sample = {}
        h2d_sig_to_fit = {}
        n_bkg_samples = {}
        n_to_fit_bkg_NC_mean_After_PSD_tccut = {}
        if self.C11_cut:
            for key in self.h2d_bkg_NC_separate.keys():
                n_to_fit_bkg_NC_mean_After_PSD_tccut[key] = n_to_fit_bkg_NC * self.dir_ratio_C11[key] * self.dir_eff_tccut[key]
                sys_uncertainty_factor = self.GetPositiveGausSysUncertainty(0.5)
                n_bkg_samples[key] = np.random.poisson(n_to_fit_bkg_NC_mean_After_PSD_tccut[key] * sys_uncertainty_factor )
        else:

            n_to_fit_bkg_NC_mean_After_PSD_tccut["Total"] = n_to_fit_bkg_NC
            sys_uncertainty_factor = self.GetPositiveGausSysUncertainty(0.5)
            n_bkg_samples["Total"] = np.random.poisson(n_to_fit_bkg_NC_mean_After_PSD_tccut["Total"]* sys_uncertainty_factor)
        # n_sig_samples = n_to_fit_sig
        # n_bkg_samples = n_to_fit_bkg_NC
        for key in self.h2d_sig.keys():
            sig_sample[key] = self.h2d_sig_full[key].sample( n_sig_samples,seed=seed)
        bkg_NC_sample = {key:self.h2d_bkg_NC_full_separate[key].sample(n_bkg_samples[key], seed=seed) for key in self.h2d_bkg_NC_separate.keys()}
        dir_other_bkg_samples = {}
        for i_key,key in enumerate(self.dir_h2d_other_bkg_full.keys()):
            np.random.seed(self.seed + 100+i_key)
            # print(key+":\t", self.dir_n_other_bkg[key])
            sys_uncertainty_factor = self.GetPositiveGausSysUncertainty(0.5)
            n_samples = np.random.poisson(self.dir_n_other_bkg[key]  * sys_uncertainty_factor)
            # n_samples = int(self.dir_n_other_bkg[key])
            if print_fit_result:
               print("input "+key+":\t", n_samples)
            dir_other_bkg_samples[key] = self.dir_h2d_other_bkg_full[key].sample( n_samples,seed=seed)
            self.dir_h2d_other_bkg_to_fit[key] = hl.hist((dir_other_bkg_samples[key][0], dir_other_bkg_samples[key][1]), bins=self.n_bins)
        # sig_sample = self.h2d_sig_full.sample(int(n_to_fit_sig), seed=seed)
        # bkg_NC_sample = self.h2d_bkg_NC_full.sample(int(n_to_fit_bkg_NC), seed=seed)
        for key in self.h2d_sig.keys():
            h2d_sig_to_fit[key] = hl.hist((sig_sample[key][0], sig_sample[key][1]), bins=self.n_bins)
        h2d_bkg_NC_to_fit = {}
        for key in self.h2d_bkg_NC_separate.keys():
            h2d_bkg_NC_to_fit[key] = hl.hist((bkg_NC_sample[key][0], bkg_NC_sample[key][1]), bins=self.n_bins)
        self.input_sig_n = 0
        for key in self.h2d_sig.keys():
            self.input_sig_n = np.sum(h2d_sig_to_fit[key].values)
        self.input_bkg_NC_n = {key:np.sum(h2d_bkg_NC_to_fit[key].values) for key in self.h2d_bkg_NC_separate.keys()}
        if print_fit_result:
            print("input DSNB:\t", n_sig_samples)
            print("input atm-NC:\t", n_bkg_samples)
            print("input atm-NC (After PSD):\t", self.input_bkg_NC_n)
            print("Ratio of C11:\t", self.dir_ratio_C11)
            for key in self.h2d_bkg_NC_separate.keys():
                print(f"Mean value of NC({key}):\t", n_to_fit_bkg_NC*self.dir_ratio_C11[key])
                print(f"Mean value of NC({key}(After PSD and tccut)", n_to_fit_bkg_NC_mean_After_PSD_tccut[key]*self.ratio_bkg_NC_After_PSD[key])
        if plot_2d_to_fit:
            fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(16,6))
            if n_to_fit_sig != 0:
                for key in self.h2d_sig.keys():
                    hl.plot2d(ax4, h2d_sig_to_fit[key], log=True, cbar=True, clabel="counts per bin")
                    SetTitle(f"Signal({key})", ax4)
            else:
                print("h2d_sig_to_fit:\t", h2d_sig_to_fit)
            for key in self.h2d_bkg_NC_separate.keys():
                hl.plot2d(ax5, h2d_bkg_NC_to_fit[key], log=True, cbar=True, clabel="counts per bin")
                SetTitle("NC", ax5)
        if plot_2d_to_fit_full_bins:
            plt.figure()
            # plt.hist2d(bkg_NC_sample[0], bkg_NC_sample[1], bins=self.n_bins_full, norm=LogNorm())
            plt.hist2d(bkg_NC_sample[0], bkg_NC_sample[1], bins=self.n_bins_full)
            plt.title("NC with full bins")
            plt.xlabel("Prediction Output")
            plt.ylabel("$E_{quen}$")
            plt.colorbar()
            plt.savefig(dir_save_fig+"NC_samples.png")
            for key in self.dir_h2d_other_bkg_full.keys():
                plt.figure()
                plt.hist2d(dir_other_bkg_samples[key][0], dir_other_bkg_samples[key][1], bins=self.n_bins_full)
                # norm=LogNorm())
                plt.title(f"{key} with full bins")
                plt.colorbar()
                plt.xlabel("Prediction Output")
                plt.ylabel("$E_{quen}$")
                plt.savefig(dir_save_fig+key+"_samples.png")

            if not fit_2d: # plot profile of the samples to fit
                plt.figure()
                plt.hist(bkg_NC_sample[1], bins=self.n_bins_full[1], histtype="step", label="atm-NC")
                for key in self.dir_h2d_other_bkg_full.keys():
                    plt.hist(dir_other_bkg_samples[key][1], bins=self.n_bins_full[1], histtype="step", label=key)
                plt.semilogy()
                plt.xlabel("$E_{quen}$")
                plt.title("1D Samples")
                plt.legend()
            plt.show()
            exit()

        self.h2d_sig_to_fit = h2d_sig_to_fit
        self.h2d_bkg_NC_to_fit = h2d_bkg_NC_to_fit
        self.h2d_to_fit = h2d_sig_to_fit["FV1"]
        if self.add_fiducial_volume_2:
            self.h2d_to_fit = h2d_sig_to_fit["FV2"]
        for key in self.h2d_bkg_NC_separate.keys():
            self.h2d_to_fit += h2d_bkg_NC_to_fit[key]
        for key in self.dir_h2d_other_bkg_to_fit.keys():
            self.h2d_to_fit = self.h2d_to_fit + self.dir_h2d_other_bkg_to_fit[key]
        # print(np.sum(h2d_sig_to_fit.values), np.sum(h2d_bkg_NC_to_fit.values), np.sum(self.h2d_to_fit.values))
        if plot_2d_to_fit:
            fig4, ax6 = plt.subplots()
            hl.plot2d(ax6,self.h2d_to_fit, log=True, cbar=True, clabel="counts per bin")
            plt.show()

    def LikelihoodFunc(self, v_n:np.ndarray):
        """

        Args:
            v_n: v_n[0] is for the Nevt of signal e.g DSNB,
             v_n[1] is for the Nevt of bkg e.g nu_atm
             v_n[2:self.n_h2d_to_fit] is for the other bkg like FastN
             v_n[self.n_h2d_to_fit:] is for epsilon of NC, ....other bkg(attention: not including DSNB)

        Returns:
            nll which are supposed to be minimized by minuit.

        """
        def LogFactorial(v_d_j:np.ndarray):
            v_to_sum = np.zeros(v_d_j.shape)
            for j in range(len(v_d_j)):
                for k in range(len(v_d_j[j])):
                    if v_d_j[j][k]>0:
                        v_to_sum[j][k] = np.sum(np.array([np.log(i) for i in range(1,int(v_d_j[j][k])+1)]))
            return v_to_sum

        # Attention: v_n[self.n_h2d_need_to_fit:] is for the epsilon of NC, .. other bkg
        v_n_epsilon = v_n[self.n_h2d_need_to_fit:]

        n_j, N_exp = 0, 0
        n_j_FV2, N_exp_FV2 = 0, 0
        for i,key in enumerate(self.h2d_sig.keys()):
            if key=="FV1":
                n_j += v_n[i]*self.h2d_sig[key].values
                N_exp += v_n[i]
            elif key=="FV2":
                n_j_FV2 += v_n[i]*self.h2d_sig[key].values
                N_exp_FV2 += v_n[i]
            else:
                print("Wrong Key, Please Check the Code!!!!!!!!!!")
                exit(0)
        for i, key in enumerate(self.h2d_bkg_NC.keys()):
            n_j += (1+v_n_epsilon[0+i])*v_n[self.n_h2d_sig+i]*self.h2d_bkg_NC[key].values
            N_exp += (1+v_n_epsilon[0+i])*v_n[self.n_h2d_sig+i]
        for i, key in enumerate(self.dir_h2d_other_bkg.keys()):
            if self.dir_is_FV2[key]:
                n_j_FV2 = n_j_FV2 + (1+v_n_epsilon[self.map_FV2_to_epsilon[key]])*v_n[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]*self.dir_h2d_other_bkg[key].values
                N_exp_FV2 = N_exp_FV2 + (1+v_n_epsilon[self.map_FV2_to_epsilon[key]])*v_n[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]
            else:
                n_j = n_j + (1+v_n_epsilon[i+len(self.h2d_bkg_NC.keys())])*v_n[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]*self.dir_h2d_other_bkg[key].values
                N_exp = N_exp + (1+v_n_epsilon[i+len(self.h2d_bkg_NC.keys())])*v_n[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]

        #set pdf = 0 as 1 in order not to encounter nan in log(pdf)
        log_n_j = np.zeros(n_j.shape)
        indices = (n_j>0)
        log_n_j[indices] = np.log(n_j[indices])

        if self.add_fiducial_volume_2:
            log_n_j_FV2 = np.zeros(n_j_FV2.shape)
            indices_FV2 = (n_j_FV2 > 0)
            log_n_j_FV2[indices_FV2] = np.log(n_j_FV2[indices_FV2])
            nll = - 2. * ( np.sum(self.h2d_to_fit.values*log_n_j -n_j-LogFactorial(self.h2d_to_fit.values)) +
                      np.sum(self.h2d_to_fit_FV2.values*log_n_j_FV2 -n_j_FV2-LogFactorial(self.h2d_to_fit_FV2.values)) )
                            # +self.h2d_to_fit.values-self.h2d_to_fit.values*np.log(self.h2d_to_fit.values))
        else:
            nll = - 2. * ( np.sum(self.h2d_to_fit.values*log_n_j -n_j-LogFactorial(self.h2d_to_fit.values)) )
                           # -LogFactorial(self.h2d_to_fit.values))
        #+ (N_exp - np.sum(self.h2d_to_fit.values)*np.log(N_exp))*2
        """
        Add the constraint of fast neutron
        """
        # nll += (v_n[5] -self.ratio_other_bkg["FastN"]*self.dir_n_other_bkg["FastN"])**2/(self.ratio_other_bkg["FastN"]*self.dir_n_other_bkg["FastN"])**2
        """
        Add systematic uncertainty for all the bkg
        """
        # nll += (v_n_epsilon[0])**2/0.5**2
        for i, key in enumerate(self.h2d_bkg_NC.keys()):
            nll += (v_n[self.n_h2d_need_to_fit+i])**2/0.5**2
        # print(v_n_epsilon**2/0.5**2)
        for i, key in enumerate(self.dir_h2d_other_bkg.keys()):
            # nll += (v_n_epsilon[i+1]/self.dir_other_bkg_sys_uncertainty[key])**2
            if not self.dir_is_FV2[key]:
                nll += (v_n[self.n_h2d_need_to_fit+i+len(self.h2d_bkg_NC.keys())]/self.dir_other_bkg_sys_uncertainty[key])**2

        # Penalize negative bins
        # indices_negative = (n_j<0)
        # nll += (-5.)*np.sum(n_j[indices_negative])

        if self.plot_v_nll:
            print("v_n:\t:", v_n)
            print(f"nll:\t{nll}\n")
            self.v_nll.append(nll)
        return nll


    def PLotDataWithErrorBar(self,h_data:hl.Hist, ax_plot:plt.Axes):
        bins_width=np.diff(h_data.bins)[0]/2
        bins_h_data=np.array(h_data. bins[0])
        bins_center=(bins_h_data[:-1]+bins_h_data[1:])/2
        bins_error=np. sqrt(h_data. values)/2
        ax_plot.errorbar(bins_center,h_data.values, xerr=bins_width, yerr=bins_error, marker="+",color="black", Ls="none", Label="Input Events")

    def PlotErrorBandWithBinEdge(self, bin_edge, error_band_lower, error_band_upper, ax:plt.Axes):
        x_return = []
        y_upper_return = []
        y_lower_return = []
        for i in range(len(bin_edge)-1):
            x_return.append(bin_edge[i])
            y_upper_return.append(error_band_upper[i])
            y_lower_return.append(error_band_lower[i])
            x_return.append(bin_edge[i+1])
            y_upper_return.append(error_band_upper[i])
            y_lower_return.append(error_band_lower[i])
        ax.fill_between(x_return, y_upper_return , y_lower_return,alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=4, linestyle='dashdot', antialiased=True)


    def PlotFitProfile(self, m, plot_FV2=False):
        """
        This function is for checking the fit result whether is correct and plot the related profile

        Args:
            m: the object of iminuit which can return the fit result

        Returns:

        """

        plot_data_to_fit_2d = True
        v_fit_once = m.np_values()
        v_fit_errors = m.np_errors()
        v_fit_error_band_upper = v_fit_once+v_fit_errors
        v_fit_error_band_lower = v_fit_once-v_fit_errors
        if not plot_FV2:
            result_fit = v_fit_once[0] * self.h2d_sig["FV1"]
            result_fit_error_upper = self.h2d_sig["FV1"].values * v_fit_error_band_upper[0]
            result_fit_error_lower = self.h2d_sig["FV1"].values * v_fit_error_band_lower[0]
            title_FV = "FV1"
        else:
            result_fit = v_fit_once[1] * self.h2d_sig["FV2"]
            result_fit_error_upper = self.h2d_sig["FV2"].values * v_fit_error_band_upper[1]
            result_fit_error_lower = self.h2d_sig["FV2"].values * v_fit_error_band_lower[1]
            title_FV = "FV2"

        print("################## Check Fit Results ################################")
        print("v_fit:\t", v_fit_once)
        print("v_fit_errors:\t", v_fit_errors)
        print("n_h2d_to_fit:\t", self.n_h2d_need_to_fit)
        print("n_h2d_sig_to_fit:\t", self.n_h2d_sig)
        print("#####################################################################")
        if not plot_FV2:
            for i, key in enumerate(self.h2d_bkg_NC.keys()):
                result_fit += (1+v_fit_once[self.n_h2d_need_to_fit+i])*v_fit_once[self.n_h2d_sig+i] * self.h2d_bkg_NC[key]
                result_fit_error_upper += ((1+v_fit_error_band_upper[self.n_h2d_need_to_fit+i])*v_fit_once[self.n_h2d_sig+i] * self.h2d_bkg_NC[key].values)
                result_fit_error_lower += ((1+v_fit_error_band_lower[self.n_h2d_need_to_fit+i])*v_fit_once[self.n_h2d_sig+i] * self.h2d_bkg_NC[key].values)

        for i,key in enumerate(self.dir_h2d_other_bkg.keys()):
            if plot_FV2 and self.dir_is_FV2[key]:
                result_fit = result_fit+(1+v_fit_once[self.n_h2d_need_to_fit+self.map_FV2_to_epsilon[key]])*v_fit_once[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]*self.dir_h2d_other_bkg[key]
                result_fit_error_upper += (1+v_fit_error_band_upper[self.n_h2d_need_to_fit+self.map_FV2_to_epsilon[key]])*v_fit_once[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]*self.dir_h2d_other_bkg[key].values
                result_fit_error_lower += (1+v_fit_error_band_lower[self.n_h2d_need_to_fit+self.map_FV2_to_epsilon[key]])*v_fit_once[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]*self.dir_h2d_other_bkg[key].values
            elif (not plot_FV2) and (not self.dir_is_FV2[key]):
                result_fit = result_fit+(1+v_fit_once[self.n_h2d_need_to_fit+i+len(self.h2d_bkg_NC.keys())])*v_fit_once[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]*self.dir_h2d_other_bkg[key]
                result_fit_error_upper += (1+v_fit_error_band_upper[self.n_h2d_need_to_fit+i+len(self.h2d_bkg_NC.keys())])*v_fit_once[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]*self.dir_h2d_other_bkg[key].values
                result_fit_error_lower += (1+v_fit_error_band_lower[self.n_h2d_need_to_fit+i+len(self.h2d_bkg_NC.keys())])*v_fit_once[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]*self.dir_h2d_other_bkg[key].values

        fig_profile_PSD, ax_profile_PSD = plt.subplots()
        fig_profile_Edep, ax_profile_Edep = plt.subplots()
        v_axes = [ax_profile_PSD, ax_profile_Edep]
        sum_axis_index = [1, 0]
        for j in range(len(v_axes)):
            hl.plot1d(v_axes[j], result_fit.project([j]), label="Fit Result")
            # print(np.sum(result_fit_error_lower, axis=1), result_fit.project([1]).bins[0])
            self.PlotErrorBandWithBinEdge(result_fit.project([j]).bins[0], np.sum(result_fit_error_lower, axis=sum_axis_index[j]) ,np.sum(result_fit_error_upper, axis=sum_axis_index[j]), v_axes[j])
            # hl.plot1d(v_axes[j], self.h2d_to_fit.project([0]), errorbars=np.ones((2,len(self.h2d_to_fit.project([0]).values))),crosses=True,label="Input Events", color="black")
            if not plot_FV2:
                hl.plot1d(v_axes[j], v_fit_once[0]*self.h2d_sig["FV1"].project([j]), ls="--" ,label=f"DSNB_FV1 Fit")
                for i, key in enumerate(self.h2d_bkg_NC.keys()):
                    if key == "Total":
                        hl.plot1d(v_axes[j],
                              v_fit_once[i + self.n_h2d_sig] * (1 + v_fit_once[i + self.n_h2d_need_to_fit]) *
                              self.h2d_bkg_NC[key].project([j]), ls="--", label=f"NC_FV1 Fit")
                    else:
                        hl.plot1d(v_axes[j],
                                  v_fit_once[i + self.n_h2d_sig] * (1 + v_fit_once[i + self.n_h2d_need_to_fit]) *
                                  self.h2d_bkg_NC[key].project([j]), ls="--", label=f"NC_FV1({key}) Fit")
            else:
                hl.plot1d(v_axes[j], v_fit_once[1]*self.h2d_sig["FV2"].project([j]), ls="--" ,label=f"DSNB_FV2 Fit")

            for i,key in enumerate(self.dir_h2d_other_bkg):
                if plot_FV2 and self.dir_is_FV2[key]:
                    hl.plot1d(v_axes[j], v_fit_once[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]*\
                          (v_fit_once[self.n_h2d_need_to_fit+self.map_FV2_to_epsilon[key]]+1)*self.dir_h2d_other_bkg[key].project([j]), ls="--" , label=key+" Fit")
                elif (not plot_FV2) and (not self.dir_is_FV2[key]):
                    hl.plot1d(v_axes[j], v_fit_once[i+self.n_h2d_sig+len(self.h2d_bkg_NC.keys())]* \
                              (v_fit_once[self.n_h2d_need_to_fit+i+len(self.h2d_bkg_NC.keys())]+1)*self.dir_h2d_other_bkg[key].project([j]), ls="--" , label=key+"_FV1 Fit")

            if not plot_FV2:
                self.PLotDataWithErrorBar(h_data=self.h2d_to_fit.project([j]),ax_plot=v_axes[j])
            else:
                self.PLotDataWithErrorBar(h_data=self.h2d_to_fit_FV2.project([j]),ax_plot=v_axes[j])
            if j==0:
                v_axes[j].set_xlabel("PSD Output")
                v_axes[j].set_xlim(result_fit.project([j]).bins[0][0], result_fit.project([j]).bins[0][-1])
                v_axes[j].set_title("Projection of PSD Output"+f"({title_FV})")
                v_axes[j].legend()
            elif j==1:
                v_axes[j].set_xlabel("$E_{quen}$")
                v_axes[j].set_xlim(result_fit.project([j]).bins[0][0], result_fit.project([j]).bins[0][-1])
                v_axes[j].set_title("Projection of $E_{quen}$"+f"({title_FV})")
                v_axes[j].legend()

        if plot_data_to_fit_2d:
            fig2_data, ax2_data = plt.subplots()
            hl.plot2d(ax2_data, result_fit, log=True, cbar=True, clabel="counts per bin")
            if not plot_FV2:
                SetTitle(f"Asimov Data To Fit (FV1)", ax2_data)
            else:
                SetTitle(f"Asimov Data To Fit (FV2)", ax2_data)

        print("chi2/ndf:\t", m.fval/((len(self.n_bins[0])-1)*(len(self.n_bins[1])-1)-2-len(self.dir_h2d_other_bkg.keys())))
        print("Errors:\t", m.np_errors)
        if not self.add_fiducial_volume_2:
            plt.show()

    def FitHistZeroFix(self, v_n_initial):
        """
        This function is for the best fitting (not fix any parameters) and getting the minimal chi2
        :param v_n_initial:
        :return:
        """
        if self.print_fit_result:
            print("v_n_initial:\t",v_n_initial)
        v_limit = [(fit_down_limit_sig, None)]*self.n_h2d_sig
        v_fix = [False]*self.n_h2d_sig
        for key in self.h2d_bkg_NC.keys():
            v_limit.append((0, None))
            v_fix.append(True)
        for i in range(len(self.dir_h2d_other_bkg.keys())):
            v_limit.append((fit_down_limit_other_bkg, fit_up_limit_other_bkg))
            v_fix.append(True)
        for i in range(len(self.dir_h2d_other_bkg.keys())+len(self.h2d_bkg_NC.keys())-self.n_key_FV2):
            v_limit.append((-eposilon_limit, None))
            v_fix.append(False)
        v_error = np.ones(len(v_limit))*0.01
        # error is for the step for minimization process, errordef=0.5 is for the max likelihood fit, errordef =1 is for the min chi2 fit
        m = Minuit.from_array_func(self.LikelihoodFunc, v_n_initial,error=v_error, limit=v_limit, errordef=0.5,fix=v_fix) #iminuit  1.5.4    pypi_0    pypi
        m.migrad()
        self.fitter = m

        if self.check_result :
            self.PlotFitProfile(m)
            if self.add_fiducial_volume_2:
                self.PlotFitProfile(m, plot_FV2=True)
                plt.show()
        return (m.np_values(), m.fval)

    def FitHistFixSigN(self, v_n_initial):
        """
        This function aims to get chi2 profile which fix the number of signal fitting
        :param v_n_initial: initial values for number of events
        :return:
        """
        if self.print_fit_result:
            print("v_n_initial:\t",v_n_initial)
        v_limit = [(fit_down_limit_sig, None) ]*self.n_h2d_sig
        v_fix = [ True ]*self.n_h2d_sig
        for key in self.h2d_bkg_NC.keys():
            v_limit.append((0, None))
            v_fix.append(True)
        for i in range(len(self.dir_h2d_other_bkg.keys())):
            v_limit.append((fit_down_limit_other_bkg, fit_up_limit_other_bkg))
            v_fix.append(True)
        for i in range(len(self.dir_h2d_other_bkg.keys())+len(self.h2d_bkg_NC.keys())-self.n_key_FV2):
            v_limit.append((-eposilon_limit, None))
            v_fix.append(False)
        v_error = np.ones(len(v_limit))*0.01
        m = Minuit.from_array_func(self.LikelihoodFunc, v_n_initial, error=v_error, limit=v_limit,\
                                   fix=v_fix, errordef=0.5 ) #iminuit       1.5.4     pypi_0    pypi
        m.migrad()
        self.fitter_fix = m
        self.fitter = self.fitter_fix
        if self.check_result:
            self.PlotFitProfile(m)
            if self.add_fiducial_volume_2:
                self.PlotFitProfile(m, plot_FV2=True)
                plt.show()
        return (m.np_values(),m.fval)

    # def FitHistFixOne(self):
def PrintFitResult(v_n):
    """
    Print the fitting results in fit vector
    :param v_n: fit vector
    :return:
    """
    print("Total input events:\t", np.sum(flh.h2d_to_fit.values))
    print("DSNB:\t", v_n[0])
    print("atmNC:\t", v_n[1])
    for i,key in enumerate(flh.dir_h2d_other_bkg.keys()):
        print(key+":\t", v_n[i+2])
        if key == "FastN" and check_FastN_input_hist:
            if v_n[i+2]>7:
                flh.PlotHistToFit()
    print("Total fit events:\t", np.sum(v_n[:-1]))
    print("fit vector:\t",v_n)
    print("-------------------------------------")


def BestFitFlow(flh, time_yr=10, no_fix_n_sig=True, n_sig_fix=(0,0)):
    ratio_time = time_yr/10
    v_val_one_nofix = []
    for i_try in range(n_try_nofix):  # try different initial values for best fitting
        # v_n_other_bkg_initial = np.random.randint(0, up_limit_random_init, size=len(flh.dir_h2d_other_bkg.keys()))
        v_n_other_bkg_initial = []
        n_DSNB_initial = []
        if align_sensitivity and not fit_2d and flh.sig_eff_81:
            n_NC_initial = [flh.input_bkg_NC_align_sensitivity*ratio_time]
            for key in flh.dir_h2d_other_bkg.keys():
                v_n_other_bkg_initial.append(flh.dir_n_other_bkg_1d_align_sensitivity[key]*ratio_time)
        else:
            if flh.C11_cut and flh.separate_C11_fitting:
                n_NC_initial = []
                for key in flh.h2d_bkg_NC.keys():
                    n_NC_initial.append( flh.n_NC_10yr * ratio_time* flh.dir_ratio_C11[key] * flh.ratio_bkg_NC_After_PSD[key]*flh.dir_eff_tccut[key])
            elif flh.C11_cut and not flh.separate_C11_fitting:
                n_NC_initial = [0]
                for key in flh.h2d_bkg_NC_separate.keys():
                    n_NC_initial[0] += ( flh.n_NC_10yr * ratio_time* flh.dir_ratio_C11[key] * flh.ratio_bkg_NC_After_PSD[key])*flh.dir_eff_tccut[key]
            else:
                n_NC_initial = [flh.ratio_bkg_NC_After_PSD["Total"] * flh.n_NC_10yr * ratio_time]
            print("################Initial values of other backgrounds ################################")
            for i,key in enumerate(flh.dir_h2d_other_bkg.keys()):
                v_n_other_bkg_initial.append(flh.dir_n_other_bkg[key]*flh.ratio_other_bkg[key]*ratio_time)
                print(f"{key}:\t", v_n_other_bkg_initial[i])
            print("###################################################################################")
        if not set_initial_epsilon_zeros:
            v_n_other_bkg_initial_epsilon = np.random.uniform(-1, 1, size=len(flh.dir_h2d_other_bkg.keys())-flh.n_key_FV2)
            if flh.C11_cut:
                NC_bkg_initial_epsilon = np.random.uniform(-1, 1, size=len(flh.h2d_bkg_NC.keys()))
            else:
                NC_bkg_initial_epsilon = [np.random.uniform(-1, 1)]
        else:
            NC_bkg_initial_epsilon = np.zeros(len(flh.h2d_bkg_NC.keys()))
            v_n_other_bkg_initial_epsilon = np.zeros(len(flh.dir_other_bkg.keys())-flh.n_key_FV2)
        if no_fix_n_sig:
            if flh.add_fiducial_volume_2:
                n_DSNB_initial = np.random.randint(0, up_limit_random_init_DSNB, size=flh.n_h2d_sig)
            else:
                n_DSNB_initial = [np.random.randint(0, up_limit_random_init_DSNB)]
            v_n_initial = np.concatenate(
                (n_DSNB_initial, n_NC_initial,
                 v_n_other_bkg_initial,
                 NC_bkg_initial_epsilon,
                 v_n_other_bkg_initial_epsilon))
            v_n_tmp, f_val_nofix_tmp = flh.FitHistZeroFix(v_n_initial=v_n_initial)
        else:
            if flh.add_fiducial_volume_2:
                n_sig_fix = list(n_sig_fix)
            else:
                n_sig_fix = [n_sig_fix]
            v_n_initial = np.concatenate((n_sig_fix, n_NC_initial,
                                                                    v_n_other_bkg_initial,
                                                                    NC_bkg_initial_epsilon,
                                                                    v_n_other_bkg_initial_epsilon))
            v_n_tmp, f_val_nofix_tmp =flh.FitHistFixSigN(v_n_initial=v_n_initial)
        # print("v_n_initial:\t", v_n_initial)
        num_iterations = 0
        while (not (flh.fitter.get_fmin()['is_valid'] and flh.fitter.get_fmin()['has_accurate_covar'])):
            if num_iterations > 9:
                break
            num_iterations += 1
            print(f"Refitting!! {num_iterations} times")
            # v_n_other_bkg_initial = np.random.randint(0, up_limit_random_init, size=len(flh.dir_h2d_other_bkg.keys()))
            if not set_initial_epsilon_zeros:
                v_n_other_bkg_initial_epsilon = np.random.uniform(-1, 1, size=len(flh.dir_h2d_other_bkg.keys()))
                if flh.C11_cut:
                    NC_bkg_initial_epsilon = np.random.uniform(-1, 1, size=2)
                else:
                    NC_bkg_initial_epsilon = [np.random.uniform(-1, 1)]
            else:
                v_n_other_bkg_initial_epsilon = np.zeros(len(flh.dir_h2d_other_bkg.keys()))
                NC_bkg_initial_epsilon = np.zeros(len(flh.h2d_bkg_NC.keys()))
            if no_fix_n_sig:
                v_n_tmp, f_val_nofix_tmp = flh.FitHistZeroFix(
                np.concatenate(
                    (list(np.random.randint(0, up_limit_random_init_DSNB, size=flh.n_h2d_sig)), n_NC_initial,
                     v_n_other_bkg_initial,
                     NC_bkg_initial_epsilon,
                     v_n_other_bkg_initial_epsilon)))
            else:
                v_n_tmp, f_val_nofix_tmp = flh.FitHistFixSigN(
                v_n_initial=np.concatenate((np.zeros((flh.n_h2d_sig)), n_NC_initial,
                                        v_n_other_bkg_initial,
                                        NC_bkg_initial_epsilon,
                                        v_n_other_bkg_initial_epsilon)))
        v_val_one_nofix.append(f_val_nofix_tmp)
        if len(v_val_one_nofix) == 1 or np.min(v_val_one_nofix[:-1]) >= f_val_nofix_tmp:
            v_n = v_n_tmp
    f_val_nofix = np.min(v_val_one_nofix)
    return (v_n, f_val_nofix)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DSNB fitter')
    parser.add_argument("--fit1d", "-f", action="store_true", help="whether use 2d fit", default=False )
    # parser.add_argument("--Nspe", "-N", type=int, default=0, help="this num is to control which ")
    args = parser.parse_args()

    ########## set some related parameters ##################
    import os
    dir_save_fig = "figure_save/"
    if not os.path.exists(dir_save_fig) or not os.path.exists("./fit_result_npz"):
        os.makedirs(dir_save_fig)
        os.makedirs("./fit_result_npz")
    fit_2d = (not  args.fit1d)
    if fit_2d:
        label_fit_method = "2d"
    else:
        label_fit_method = "1d"
    label_version = "v1_TMVA"
    fit_Asimov_dataset = True
    save_2d_PDF = False
    print_fit_result = False
    # fit_2d = False
    set_initial_epsilon_zeros = True
    plot_nll_transition = False
    without_normalize = False
    plot_pdf_projection = False

    only_best_fit = False #boolean switch for whether doing the best fit not get the chi2 profile
    if only_best_fit:
        plot_chi2 = False
    else:
        plot_chi2 = True
    uncertainty_other_bkg = 0.5
    fit_down_limit_sig = 0
    align_sensitivity = False
    use_additive_DSNB_And_NC = False

    fit_down_limit_other_bkg = 0
    fit_up_limit_other_bkg = 40
    up_limit_random_init = 20
    up_limit_random_init_DSNB = 10
    up_limit_random_init_NC = 10
    eposilon_limit = 1
    nll_last_fix = 0
    # set how many times to try the initial values
    if fit_2d:
        n_try_fix = 1
        n_try_nofix = 1
    else:
        n_try_fix = 1
        n_try_nofix = 1
    dir_other_bkg = {}

    ###########Number of Input Events######################
    n_CC_FV1 = 1.9
    n_CC_FV2 = 0.4
    n_FastN_FV1 = 11.7
    n_FastN_FV2 = 29.2
    n_NC_FV1 = 405.7
    n_NC_FV2 = 95
    #######################################################


    flh = FLH()

    if flh.use_sk_data:
        name_file_predict = "/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_time_jobs_DSNB_sk_data/predict_0.npz"
    else:
        name_file_predict = "/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_TMVA/predict_0_within16m.npz"

    flh.plot_v_nll = plot_nll_transition

    if flh.C11_cut:
        label_version += "_add_tccut"
    if flh.add_fiducial_volume_2:
        label_version += "_add_FV2"

    # Load the data about PSD and Equen
    flh.LoadPrediction(name_file_predict)
    name_fiducial_volume = ""
    name_fiducial_volume_2 = ""
    if flh.use_sk_data:
        name_dir_other_bkg = "/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_time_jobs_DSNB_sk_data/"
    else:
        name_dir_other_bkg = "/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_TMVA/"
        name_fiducial_volume = "_within16m"
        name_fiducial_volume_2 = "_outside16m"
    flh.LoadOtherBkg(name_dir_other_bkg+"atm-CC_samples_predict_0"+name_fiducial_volume+".npz",
                     key="dict_samples",
                     key_here="CC",
                     sys_uncertainty=uncertainty_other_bkg, n_events_10yr=n_CC_FV1)

    flh.SetN_Bins()
    if flh.n_bins[1][0]<11.5: # if n_bins' not include 10-12MeV, anti_nu from Reactor and Li9He8 will not be the background.
        flh.LoadOtherBkg(name_dir_other_bkg+"Reactor-anti-Nu_samples_predict_0"+name_fiducial_volume+".npz",
                         key="dict_samples",
                         key_here="Reactor-anti-Nu",
                         sys_uncertainty=uncertainty_other_bkg, n_events_10yr=3.4)
        flh.LoadOtherBkg(name_dir_other_bkg+"Li9He8_samples_predict_0"+name_fiducial_volume+".npz",
                         key="dict_samples",
                         key_here="He8Li9",
                         sys_uncertainty=uncertainty_other_bkg, n_events_10yr=0.08)
    if flh.use_sk_data:
        flh.LoadOtherBkg(name_dir_other_bkg+"FastN_samples_predict_0.npz",
                         key="dict_samples",
                         key_here="FastN",
                         sys_uncertainty=uncertainty_other_bkg, n_events_10yr=n_FastN_FV1)
    else:
        flh.LoadOtherBkg(name_dir_other_bkg+"FastN_samples_predict_0"+name_fiducial_volume+"_C11.npz",
                         key="dict_samples",
                         key_here="FastN",
                         sys_uncertainty=uncertainty_other_bkg, n_events_10yr=n_FastN_FV1)
        flh.LoadNCOtherModel(name_dir_other_bkg+"NC_Average_Model_samples_predict_0"+name_fiducial_volume+".npz",
                         key_in_file="dict_samples",
                         key_here="Average_Model")

    if flh.add_fiducial_volume_2:
        flh.LoadOtherBkg(name_dir_other_bkg+"FastN_samples_predict_0"+name_fiducial_volume_2+"_C11.npz",
                         key="dict_samples",
                         key_here="FastN_FV2",
                         sys_uncertainty=uncertainty_other_bkg, n_events_10yr=n_FastN_FV2, is_FV2=True)
        flh.LoadOtherBkg(name_dir_other_bkg+"atm-CC_samples_predict_0"+name_fiducial_volume_2+".npz",
                     key="dict_samples",
                     key_here="CC_FV2",
                     sys_uncertainty=uncertainty_other_bkg, n_events_10yr=n_CC_FV2, is_FV2=True)
        flh.LoadFV2NCAndDSNB(name_file="/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_TMVA/predict_0"+name_fiducial_volume_2+".npz",
                             NC_uncertainty=0.5, n_NC_FV2=n_NC_FV2)
        if flh.n_bins[1][0]<11.5: # if n_bins' not include 10-12MeV, anti_nu from Reactor and Li9He8 will not be the background.
            print("The rate of He8Li9 and Nu_Reactor in FV2 are not corrected. So Changing the rate is needed!!!!")
            exit(0)
            flh.LoadOtherBkg(name_dir_other_bkg + "Reactor-anti-Nu_samples_predict_0" + name_fiducial_volume_2 + ".npz",
                             key="dict_samples",
                             key_here="Reactor-anti-Nu_FV2",
                             sys_uncertainty=uncertainty_other_bkg, n_events_10yr=3.4, is_FV2=True)
            flh.LoadOtherBkg(name_dir_other_bkg + "Li9He8_samples_predict_0" + name_fiducial_volume_2 + ".npz",
                             key="dict_samples",
                             key_here="He8Li9_FV2",
                             sys_uncertainty=uncertainty_other_bkg, n_events_10yr=0.08, is_FV2=True)


    flh.LoadDSNBOtherModel(name_dir_other_bkg+"DSNB_15MeV_samples_predict_0"+name_fiducial_volume+".npz",
                           key_in_file="dict_samples",
                           key_here="15MeV")



    # Set the events number for different kinds of events
    n_to_fit_bkg_NC = n_NC_FV1
    n_to_fit_sig = 0
    dir_n_DSNB_expected_10yr = {"15MeV_FV1":19.4, "15MeV_FV2":4.6}
    # dir_n_other_bkg = {"CC":2.4,  "FastN":9.7, "He8Li9":0.08, "Reactor-anti-Nu":3.4, }
    # flh.SetNOtherBkg(dir_n_other_bkg)
    flh.SetNDiffEDSNB(dir_n_DSNB_expected_10yr)
    flh.SetNCNumber(n_to_fit_bkg_NC)
    # flh.GetBestSNRatio()

    # Get the PDF distribution
    flh.Get2DPDFHist(fit_2d=fit_2d)

    flh.GetMapFV2ToFV1Epsilon()

    # Set 1D PDF To check Sensitivity
    if align_sensitivity and not fit_2d:
        flh.Set1DPDFToCheckSensitivity()

    # Plot the projection of PDFs
    if plot_pdf_projection:
        flh.PlotPDFProfile()

    if not fit_Asimov_dataset:
        # Preparation for getting chi2 profile
        from scipy.interpolate import interp1d
        v_uplimit_n_sig = []
        n_max_sig = 20
        time_yr = 10
        ratio_time = time_yr/10
        chi2_criteria = 2.706
        check_FastN_input_hist = False
        v_n_sig_to_fix = np.arange(0, n_max_sig+1)
        v_n_sig_to_fix_eff_correction = v_n_sig_to_fix/flh.ratio_sig
        v_fit_result = {"sig": [], "bkg": []}
        for key in flh.dir_h2d_other_bkg.keys():
            v_fit_result[key] = []
        v2D_chi2 = [] # dimension 0 is for the times of trying , dimension 1 is for the number of fix signal
        v_chi2_ndf_bestfit = []

        n_trials = 1000
        if only_best_fit:
            n_trials *= 2
        for i in trange(n_trials):
            np.random.seed(i)
            random.seed(i)
            if print_fit_result:
                print("---------------------------")
                print("seed number:\t", i)
            num_iterations = 0
            if i %100 ==0:
                print(f"Processing {i} times fitting")
            # Get the toy data histogram to fit
            flh.GetHistToFit(n_to_fit_bkg_NC=round(n_to_fit_bkg_NC), n_to_fit_sig=n_to_fit_sig, seed=i, print_fit_result=print_fit_result)

            v_val_one_nofix = [] # store nll values for different initial values and then get the min nll as fit value
            for i_try in range(n_try_nofix): # try different initial values for best fitting
                # v_n_other_bkg_initial = np.random.randint(0, up_limit_random_init, size=len(flh.dir_h2d_other_bkg.keys()))

                (v_n_tmp, f_val_nofix_tmp) = BestFitFlow(flh, time_yr=time_yr)
                v_val_one_nofix.append(f_val_nofix_tmp)
                if len(v_val_one_nofix)==1 or np.min(v_val_one_nofix[:-1]) >= f_val_nofix_tmp:
                    v_n = v_n_tmp

            # summary the best fitting result
            v_fit_result["sig"].append(v_n[0])
            v_fit_result["bkg"].append(v_n[1])
            if print_fit_result:
                PrintFitResult(v_n)
            for i, key in enumerate(flh.dir_h2d_other_bkg.keys()):
                v_fit_result[key].append(v_n[i + 2])
            f_val_nofix = np.min(v_val_one_nofix)
            v_chi2_ndf_bestfit.append(f_val_nofix/((len(flh.n_bins[0])-1)*(len(flh.n_bins[1])-1)-2-len(flh.dir_h2d_other_bkg.keys())))
            nll_last_fix = f_val_nofix

            # This fit is for getting chi2 profile
            if not only_best_fit:
                v_chi2 = []
                for j in v_n_sig_to_fix:
                    v_val_one_fix = []
                    for i_try in range(n_try_fix): # Trying different initial values
                        # v_n_other_bkg_initial = np.random.randint(0, up_limit_random_init, size=len(flh.dir_h2d_other_bkg.keys()))
                        if flh.add_fiducial_volume_2:
                            n_sig_fix = (j, 0)
                        else:
                            n_sig_fix =j
                        (v_n_fix, f_val_fix) = BestFitFlow(flh, no_fix_n_sig=False,time_yr=time_yr, n_sig_fix=n_sig_fix)

                        if plot_nll_transition:
                            plt.plot(flh.v_nll)
                            plt.show()

                        if print_fit_result:
                            print(f"---------fix result(N_sig={j}) --------------")
                            PrintFitResult(v_n_fix)
                        v_val_one_fix.append(f_val_fix)
                    # if Chi2Smooth(nll_last_fix, f_val_fix):
                    min_val_one_fix = np.min(v_val_one_fix)
                    v_chi2.append(min_val_one_fix-f_val_nofix)
                    nll_last_fix = min_val_one_fix

                index_min = np.argmin(v_chi2)
                v2D_chi2.append(v_chi2)

                # Getting the chi2 profile and chi2=2.70's point of intersection
                try:
                    f = interp1d( v_chi2[index_min:], v_n_sig_to_fix_eff_correction[index_min:], kind="linear", fill_value="extrapolate")
                    uplimit_n_sig = f(chi2_criteria)
                    v_uplimit_n_sig.append(uplimit_n_sig)
                except Exception:
                    print("Getting chi2 uplimit failed!")
                    continue

        # Plot the figure we need
        plt.figure()
        if n_to_fit_sig != 0:
            plt.hist(v_fit_result["sig"], histtype="step", bins=50)
        else:
            plt.hist(v_fit_result["sig"], histtype="step", bins=100)
        plt.xlabel("N of Signal")

        plt.figure()
        plt.hist(v_fit_result["bkg"], histtype="step", bins=100)
        plt.xlabel("N of NC")

        for i, key in enumerate(flh.dir_h2d_other_bkg.keys()):
            plt.figure()
            plt.hist(v_fit_result[key], histtype="step", bins=100)
            plt.xlabel("N of "+key)

        if plot_chi2 or not only_best_fit:
            plt.figure()
            for i in range(len(v2D_chi2)):
                plt.plot(v_n_sig_to_fix_eff_correction,v2D_chi2[i])
            plt.plot([0, n_max_sig], [chi2_criteria, chi2_criteria],"--", label="90% confidence")
            plt.ylim(0,10)
            plt.xlabel("Number of Signal Counts")
            plt.ylabel("$NLL_{min}^{fix}-NLL_{min}^{nofix}$")
            plt.legend()

            plt.savefig(dir_save_fig+"chi_profile_"+label_fit_method+f"_{label_version}.png")

            plt.figure()
            v_uplimit_n_sig = np.array(v_uplimit_n_sig).reshape(-1)
            # print(v_uplimit_n_sig)
            h_uplimit = plt.hist(v_uplimit_n_sig, histtype="step", bins=10)
            median_uplimit = np.median(v_uplimit_n_sig)
            print("median:\t", median_uplimit)
            plt.plot([median_uplimit, median_uplimit], [0, np.max(h_uplimit[0])], "--", label="median:  {:.2f}".format(median_uplimit))
            plt.xlabel("Uplimit of Number of signal")
            plt.legend()

            plt.figure()
            plt.hist(v_chi2_ndf_bestfit, bins=np.arange(0, 3, 0.05))
            plt.xlabel("$\chi^2$/ndf")

            plt.savefig(dir_save_fig+"uplimit_"+label_fit_method+"_"+label_version+".png")
            np.save(f"./fit_result_npz/v_uplimit_{label_fit_method}_{label_version}.npy", v_uplimit_n_sig)
            np.savez(f"./fit_result_npz/v_chi2_{label_fit_method}_{label_version}.npz", v_n_sig=v_n_sig_to_fix, v_chi2=v2D_chi2, ratio_sig=flh.ratio_sig["FV1"], chi2_ndf=v_chi2_ndf_bestfit)
        # plt.show()
    else:
        v_sensitivity = []
        v_year = []
        # for i_year in [10]:
        for i_year in range(1, 11):
            flh.GetAsimovDatasetToFit(time_scale_year=i_year)
            print("#############Add Signal PDF###################")
            (v_n_add_signal, f_val_nofix_add_signal) = BestFitFlow(flh, time_yr=i_year)
            print("fit Asimov Dataset nll values:\t", f_val_nofix_add_signal)
            print("keys:\t", flh.dir_h2d_other_bkg.keys())
            if align_sensitivity and not fit_2d and flh.sig_eff_81:
                print("other bkgs:\t",flh.dir_n_other_bkg_1d_align_sensitivity )
            print("fit parameters:\t", v_n_add_signal)
            print("#############Without Signal PDF###################")
            if flh.add_fiducial_volume_2:
                (v_n_without_signal, f_val_nofix_without_signal) = BestFitFlow(flh, no_fix_n_sig=False, time_yr=i_year, n_sig_fix=(0,0))
            else:
                (v_n_without_signal, f_val_nofix_without_signal) = BestFitFlow(flh, no_fix_n_sig=False, time_yr=i_year,
                                                                           n_sig_fix=0)

            print("fit Asimov Dataset nll values(no signal PDF):\t", f_val_nofix_without_signal)
            print("keys:\t", flh.dir_h2d_other_bkg.keys())
            print("fit parameters:\t", v_n_without_signal)
            sensitivity = np.sqrt(f_val_nofix_without_signal-f_val_nofix_add_signal)
            print("Sensitivity:\t", sensitivity )
            # print("Delta Sensitivity:\t", sensitivity-3.678)
            v_sensitivity.append(sensitivity)
            v_year.append(i_year)
        plt.plot(v_year, v_sensitivity)
        print("Sensitivity:\t", v_sensitivity)
        plt.xlabel("exposure [ 14.7 kt $*$ yr ] ")
        plt.ylabel("Sensitivity[$\sigma$]")
        plt.ylim(0, 6)
        if fit_2d:
            plt.title("2D Fitting")
        else:
            plt.title("1D Fitting")
        np.savez(f"./fit_result_npz/discover_potential_{label_fit_method}_{label_version}.npz", v_year=v_year, v_sensitivity=v_sensitivity)
        # plt.show()


### dictionary map : /junofs_500G/sk_psd_DSNB
###or map : /gpu_500G/LS_ML_CNN

