# -*- coding:utf-8 -*-
# @Time: 2021/5/8 16:20
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitWithComponent.py
import matplotlib.pylab as plt
from torch import nn, optim
import argparse
import numpy as np
import uproot as up
import glob
from iminuit import Minuit
import tqdm
from scipy import interpolate
from matplotlib.colors import LogNorm
import sys
import torch
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
from copy import copy
from torch.utils.data import DataLoader,Dataset
from CNNDataset import CNNDataset
from CNN_GetGammaRatio import CNN1D, CNN1D_2, Net, LinearNet
import os
import random
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

device = 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def LossFunc( y_true, y_pred):
    return torch.sum((y_true - y_pred) ** 2/( (y_pred-0.2033)**2+0.001))

def GetBinCenter(bins:np.ndarray):
    return (bins[1:]+bins[:-1])/2

def ElementsAreTheSameInList(v:list):
    return (v.count(v[0])==len(v))

def NormProfile(h_time:np.ndarray, h_edges:np.ndarray, divide_bin_width=True):
    if divide_bin_width:
        h_time_divide_width = h_time/np.diff(h_edges)
    else:
        h_time_divide_width = h_time
    return np.array(h_time_divide_width/np.max(h_time_divide_width), dtype=float)

def NormProfileByArea(h_time:np.ndarray, h_edges:np.ndarray, divide_bin_width=True):
    if divide_bin_width:
        h_time_divide_width = h_time/np.diff(h_edges)
    else:
        h_time_divide_width = h_time
    return np.array(h_time_divide_width/np.sum(h_time_divide_width*np.diff(h_edges)), dtype=float)

def NormPDFByArea(h_pdf:np.ndarray, h_edges:np.ndarray):
    return np.array(h_pdf/np.sum(h_pdf*np.diff(h_edges)), dtype=float)

def NormPDFByMax(h_pdf:np.ndarray, h_edges:np.ndarray):
    return np.array(h_pdf/np.max(h_pdf), dtype=float)

def LogFactorial(v_d_j:np.ndarray):
    v_to_sum = np.zeros(v_d_j.shape)
    for j in range(len(v_d_j)):
        for k in range(len(v_d_j[j])):
            if v_d_j[j][k]>0:
                v_to_sum[j][k] = np.sum(np.array([np.log(i) for i in range(1,int(v_d_j[j][k])+1)]))
    return v_to_sum
def GetAdjustBinsIndex(n_bins:np.ndarray, bin_start:float, bin_end:float):
    index = (n_bins>=bin_start) & (n_bins<=bin_end)
    return index


class FitWithComponent:
    def __init__(self, model_mode=None, validate_mode=False):
        self.key_no_charge = "w/o charge"
        self.key_with_charge = "w/ charge"
        self.dir_pdf_time_profile_no_charge = {}
        self.dir_pdf_time_profile_with_charge = {}
        self.dir_pdf_time_profile_no_charge_before_cut = {}
        self.dir_pdf_time_profile_with_charge_before_cut = {}
        self.dir_pdf = {}
        self.dir_pdf_before_cut = {}
        self.bins = {}
        self.bins_before_cut = {}
        self.bins_center = {}
        self.bins_center_before_cut = {}
        self.evts_to_fit = {}
        self.evts_to_fit_train = {}
        self.evts_to_fit_test = {}
        self.evts_to_fit_validate = {}
        self.h_to_fit = np.array([])
        self.fit_result_time_profile = np.array([])
        self.v_ratio_gamma_fit = []
        self.fit_profile_separate = {}
        self.n_amplitude_fit_parameters = 0
        self.n_time_parameters = 0
        self.key_pdf = ""
        self.bin_start = -20
        self.bin_end = 250
        self.upper_limit_amplitude = 2
        self.limit_time = 0

        self.not_skip_with_charge_in_normalize_data = True
        self.divide_bin_width = False
        self.norm_pdf_by_area = False
        self.check_adjust_bins = False

        if not validate_mode:
            self.model_mode = args.model
        else:
            self.model_mode = model_mode

        self.model_label = f"_{self.model_mode}"
        if not validate_mode:
            if not center_fitting:
                self.model_label+= "_uniform"
            else:
                self.model_label+="_center"
            if use_truth_time:
                self.model_label += "_truth"

        self.training_with_charge = False
        self.training_combine = False
        self.training_add_vertex = True
        self.training_add_energy = True
        self.training_with_equen_cut = True

        if not self.training_combine:
            self.training_add_vertex = False
            print("Warning!!! training_add_vertex option only avaliable for training_vertex, so we turn it off")

        if not self.training_add_vertex:
            self.training_add_energy = False
        self.save_ML_output = True
        self.load_uniform_R_lepton_data = True

        if not validate_mode:
            self.option_h_time = "" # Whether use Truth
            if use_truth_time:
                self.option_h_time = "_truth"

        if self.training_combine:
            self.model_label += "_combine"
        if self.training_add_vertex:
            self.model_label += "_add_vertex"
        if self.training_add_energy:
            self.model_label += "_add_energy"


    def LoadTimeProfilePDF(self, dir_name_files:dict):
        key_time_in_root = "h_time_average"
        key_time_with_charge_in_root ="h_time_with_charge_average"
        for key, name_file in dir_name_files.items():
            f = up.open(name_file)
            h_time, h_edges = f[key_time_in_root].to_numpy()
            h_time_with_charge, h_edges_with_charge = f[key_time_with_charge_in_root].to_numpy()
            if self.norm_pdf_by_area:
                h_time = NormPDFByArea( h_time, h_edges)
                h_time_with_charge = NormPDFByArea( h_time_with_charge, h_edges_with_charge )
            else:
                h_time = NormPDFByMax( h_time, h_edges)
                h_time_with_charge = NormPDFByMax( h_time_with_charge, h_edges_with_charge )

            self.index_h_edges = GetAdjustBinsIndex(h_edges[1:], self.bin_start, self.bin_end )
            self.index_h_edges_with_charge = GetAdjustBinsIndex(h_edges_with_charge[1:], self.bin_start, self.bin_end)
            self.index_h_time =  GetAdjustBinsIndex(GetBinCenter(h_edges[1:]), self.bin_start, self.bin_end )
            self.index_h_time_with_charge = GetAdjustBinsIndex(GetBinCenter(h_edges_with_charge[1:]), self.bin_start, self.bin_end)
            self.dir_pdf_time_profile_with_charge_before_cut[key] = h_time_with_charge[1:] # Because there is an extra bin when we generate the pdfs which is not the same as the data(h_time) to fit
            self.dir_pdf_time_profile_no_charge_before_cut[key] = h_time[1:]
            self.dir_pdf_time_profile_with_charge[key] = h_time_with_charge[1:][self.index_h_time] # Because there is an extra bin when we generate the pdfs which is not the same as the data(h_time) to fit
            self.dir_pdf_time_profile_no_charge[key] = h_time[1:][self.index_h_time_with_charge]
            # print(key,":\t", h_time)
            self.bins_before_cut[self.key_no_charge] = h_edges[1:]
            self.bins_before_cut[self.key_with_charge] = h_edges_with_charge[1:]
            self.bins[self.key_no_charge] = h_edges[1:][self.index_h_edges]
            self.bins[self.key_with_charge] = h_edges_with_charge[1:][self.index_h_edges_with_charge]
            f.close()
        
        self.dir_pdf[self.key_no_charge] = self.dir_pdf_time_profile_no_charge
        self.dir_pdf[self.key_with_charge] = self.dir_pdf_time_profile_with_charge
        self.dir_pdf_before_cut[self.key_no_charge] = self.dir_pdf_time_profile_no_charge_before_cut
        self.dir_pdf_before_cut[self.key_with_charge] = self.dir_pdf_time_profile_with_charge_before_cut
        # print("Bins:\t", self.bins)
        for key in self.bins:
            self.bins_center[key] = GetBinCenter(self.bins[key])
            self.bins_center_before_cut[key] = GetBinCenter(self.bins_before_cut[key])
        self.index_gamma = list(self.dir_pdf_time_profile_no_charge.keys()).index("gamma")
        self.n_amplitude_fit_parameters = (len(self.dir_pdf.keys())-1 if constrain_fitting_amplitude_sum_as_1 else len(self.dir_pdf_time_profile_no_charge.keys()))

        self.n_time_parameters = len(self.dir_pdf.keys())

        if fitting_with_charge:
            self.key_pdf = self.key_with_charge
        else:
            self.key_pdf = self.key_no_charge

        if self.check_adjust_bins:
            print("Check Bins Cut:\t", f"{self.bin_start}-{self.bin_end}", self.bins)
            for key_particle in self.dir_pdf[self.key_no_charge].keys():
                plt.figure()
                for j, key in enumerate(self.bins.keys()):
                    plt.step(self.bins_center_before_cut[key], self.dir_pdf_before_cut[key][key_particle])
                    plt.step(self.bins_center[key], self.dir_pdf[key][key_particle], ls="--")
                plt.title(key_particle)
            plt.show()
            exit()

    def NormTimeProfileToFit(self):
        self.evts_to_fit["h_time_norm"] = []
        self.evts_to_fit["h_time_with_charge_norm"] = []
        self.evts_to_fit["statistical_error_time"] = []
        self.evts_to_fit["statistical_error_time_with_charge"] = []
        for i in range(len(self.evts_to_fit["h_time"+self.option_h_time])):
            if self.norm_pdf_by_area:
                self.evts_to_fit["h_time_norm"].append( NormProfileByArea(h_time=self.evts_to_fit["h_time"+self.option_h_time][i][self.index_h_time], h_edges=self.bins[self.key_no_charge],
                                                                          divide_bin_width=self.divide_bin_width) )
                if self.not_skip_with_charge_in_normalize_data:
                    self.evts_to_fit["h_time_with_charge_norm"].append(NormProfileByArea(h_time=self.evts_to_fit["h_time_with_charge"+self.option_h_time][i][self.index_h_time_with_charge],
                                                                                         h_edges=self.bins[self.key_with_charge], divide_bin_width=self.divide_bin_width))
                if not use_relative_error:
                    self.evts_to_fit["statistical_error_time"].append( (self.evts_to_fit["h_time"+self.option_h_time][i][self.index_h_time])**0.5/np.sum(self.evts_to_fit["h_time"+self.option_h_time][i]) )
                    if self.not_skip_with_charge_in_normalize_data:
                        self.evts_to_fit["statistical_error_time_with_charge"].append(
                        (self.evts_to_fit["h_time_with_charge"+self.option_h_time][i][self.index_h_time_with_charge])**0.5/np.sum(self.evts_to_fit["h_time"+self.option_h_time][i]) )
                else:
                    self.evts_to_fit["statistical_error_time"].append(
                        (self.evts_to_fit["h_time"+self.option_h_time][i][self.index_h_time]) ** 0.5 / (
                            self.evts_to_fit["h_time"+self.option_h_time][i][self.index_h_time]))
                    if self.not_skip_with_charge_in_normalize_data:
                        self.evts_to_fit["statistical_error_time_with_charge"].append(
                        (self.evts_to_fit["h_time_with_charge"+self.option_h_time][i][self.index_h_time_with_charge]) ** 0.5 / (
                        self.evts_to_fit["h_time"+self.option_h_time][i][self.index_h_time_with_charge]))
            else:
                self.evts_to_fit["h_time_norm"].append( NormProfile(h_time=self.evts_to_fit["h_time"+self.option_h_time][i][self.index_h_time],
                                                                    h_edges=self.bins[self.key_no_charge], divide_bin_width=self.divide_bin_width) )
                if self.not_skip_with_charge_in_normalize_data:
                    self.evts_to_fit["h_time_with_charge_norm"].append(NormProfile(h_time=self.evts_to_fit["h_time_with_charge"+self.option_h_time][i][self.index_h_time_with_charge],
                                                                                   h_edges=self.bins[self.key_with_charge], divide_bin_width=self.divide_bin_width))
                if not use_relative_error:
                    self.evts_to_fit["statistical_error_time"].append(
                        (self.evts_to_fit["h_time"+self.option_h_time][i][self.index_h_time]) ** 0.5 / np.max(
                            self.evts_to_fit["h_time"+self.option_h_time][i]))
                    if self.not_skip_with_charge_in_normalize_data:
                        self.evts_to_fit["statistical_error_time_with_charge"].append(
                    (self.evts_to_fit["h_time_with_charge"+self.option_h_time][i][self.index_h_time_with_charge])**0.5/np.max(self.evts_to_fit["h_time"+self.option_h_time][i]) )
                else:
                    self.evts_to_fit["statistical_error_time"].append(
                        (self.evts_to_fit["h_time"+self.option_h_time][i][self.index_h_time]) ** 0.5 / (
                            self.evts_to_fit["h_time"+self.option_h_time][i][self.index_h_time]))
                    if self.not_skip_with_charge_in_normalize_data:
                        self.evts_to_fit["statistical_error_time_with_charge"].append(
                    (self.evts_to_fit["h_time_with_charge"+self.option_h_time][i][self.index_h_time_with_charge])**0.5/(self.evts_to_fit["h_time"+self.option_h_time][i][self.index_h_time_with_charge]) )

        self.evts_to_fit["h_time_norm"] = np.array(self.evts_to_fit["h_time_norm"])
        if self.not_skip_with_charge_in_normalize_data:
            self.evts_to_fit["h_time_with_charge_norm"] = np.array(self.evts_to_fit["h_time_with_charge_norm"])

    def GetFileListToFit(self, name_source:str="Neutron", short_file_list:bool=False):
        if use_ML_method:
            path_data_predict = "/afs/ihep.ac.cn/users/l/luoxj/gpu_500G/DSNB_component_fitting/"
        else:
            path_data_predict = "/afs/ihep.ac.cn/users/l/luoxj/sk_psd/"
        if name_source == "Neutron":
            self.files_list_to_fit = glob.glob(path_data_predict+"predict_withpdgdep/predict_*.npz")
        else:
            self.files_list_to_fit = glob.glob(
                    path_data_predict+f"predict_withpdgdep_{name_source}/predict_*.npz")
        print("One of Loaded Files:\t", self.files_list_to_fit[0])
        if short_file_list:
            self.files_list_to_fit = self.files_list_to_fit[:300]

    def GetHistExpected(self, v_n):
        v_time_parameters = v_n[self.n_amplitude_fit_parameters:self.n_amplitude_fit_parameters+self.n_time_parameters]
        n_exp_j = np.array([])
        for i, key in enumerate(self.dir_pdf[self.key_pdf]):
            if i == 0:
                n_exp_j = np.zeros(self.dir_pdf[self.key_pdf][key].shape)
            if constrain_fitting_amplitude_sum_as_1:
                if i == self.n_amplitude_fit_parameters:
                    n_exp_j += (1 - np.sum(v_n[:self.n_amplitude_fit_parameters])) * self.dir_pdfs_function[key](
                        self.bins_center[self.key_pdf] + v_time_parameters[i])
                else:
                    n_exp_j += v_n[i] * self.dir_pdfs_function[key](self.bins_center[self.key_pdf] + v_time_parameters[i])
            else:
                n_exp_j += v_n[i] * self.dir_pdfs_function[key](self.bins_center[self.key_pdf] + v_time_parameters[i])

        return n_exp_j

    def LikelihoodFunc(self, v_n:np.ndarray):
        n_exp_j = self.GetHistExpected(v_n)

        #set pdf = 0 as 1 in order not to encounter nan in log(pdf)
        log_n_exp_j = np.zeros(n_exp_j.shape)
        indices = (n_exp_j>0)
        log_n_exp_j[indices] = np.log(n_exp_j[indices])

        nll = - 2. * (np.sum(self.h_to_fit * log_n_exp_j - n_exp_j - LogFactorial(np.array([self.h_to_fit]))[0]))

        return nll
    def Chi2Func(self, v_n:np.ndarray):
        n_exp_j = self.GetHistExpected(v_n)
        index_non_zero_error = (self.statistical_errors_to_fit!=0.)
        if add_error_in_chi2_fitting:
            v_chi2 = np.absolute((n_exp_j[index_non_zero_error]-self.h_to_fit[index_non_zero_error])/self.statistical_errors_to_fit[index_non_zero_error])
            # v_chi2 = ((n_exp_j[index_non_zero_error]-self.h_to_fit[index_non_zero_error])/self.statistical_errors_to_fit[index_non_zero_error])**2
        else:
            v_chi2 = np.absolute(n_exp_j[index_non_zero_error] - self.h_to_fit[index_non_zero_error])
        v_chi2 = np.nan_to_num(v_chi2)
        chi2 = np.sum(v_chi2)
        # print("delta:\t", np.sum(np.absolute(n_exp_j[index_non_zero_error]-self.h_to_fit[index_non_zero_error])))
        # print("statistical errors:\t", self.statistical_errors_to_fit)
        # print("v_chi2:\t",v_chi2)
        # print("Chi2:\t", chi2)
        return chi2


    def LoadDatToFit(self):
        check_data_loaded = True
        if not self.load_uniform_R_lepton_data:
            f = np.load(self.files_list_to_fit[0],
                        allow_pickle=True)
            evts_0 = f["dir_events"].item()
            print("Loaded Data Keys:\t", evts_0.keys())
            evts = {}
            for key in evts_0.keys():
                if use_truth_time and (key == "PSD" or key == "PSD_with_charge"):
                    continue
                if not use_truth_time and (key == "h_time_with_charge_truth" or key == "h_time_truth"):
                    continue
                evts[key] = []
            for file in self.files_list_to_fit:
                with np.load(file, allow_pickle=True) as f:
                    evts_load = f["dir_events"].item()
                    for key in evts.keys():
                        if use_truth_time and (key == "PSD" or key == "PSD_with_charge"):
                            continue
                        evts[key].extend(evts_load[key])
            for key in evts.keys():
                try:
                    evts[key] = np.array(evts[key])
                except Exception:
                    continue
            self.evts_to_fit = evts
        else:
            dir_name = os.path.dirname(self.files_list_to_fit[0])
            if not center_fitting:
                f = np.load(f"{dir_name}/uniform_predict.npz", allow_pickle=True)
            else:
                f = np.load(f"{dir_name}/center_predict.npz", allow_pickle=True)
            self.evts_to_fit = f["dir_events"].item()
        self.entries = len(self.evts_to_fit["h_time"])
        if check_data_loaded:
            v_n_evts = []
            print("##############Check Number of Events in dir_evts####################")
            for key in self.evts_to_fit.keys():
                print(f"{key}:\t{len(self.evts_to_fit[key])}")
                if key != "edep":
                    v_n_evts.append(len(self.evts_to_fit[key]))
            if not ElementsAreTheSameInList(v_n_evts):
                print("The number of evts in dir of each key are not the same!!!! Check Loading Data to fit!!!")
                exit(1)
            print("####################################################################")
        self.NormTimeProfileToFit()
        self.GetVertexR()

        if plot_data_loaded:
            import sys
            sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
            from PlotMultiHistToH2D import TurnMultiHistToH2D,PlotHist2D
            if fitting_with_charge==True:
                v_plot_key = ["h_time_with_charge_norm"]
            else:
                v_plot_key = ["h_time_norm"]
            v_colors = ["orange", "red"]
            y_max = (0.1 if self.norm_pdf_by_area else 1.)
            for key in v_plot_key:
                fig=plt.figure()
                (h2d, xedges, yedges) = TurnMultiHistToH2D(v_hist=self.evts_to_fit[key], h_edges=self.bins[self.key_no_charge], up_ylimit=y_max)
                PlotHist2D(h2d, xedges, yedges, log=True, fig=fig)
                for i,key_particle in enumerate(self.dir_pdfs_function.keys()):
                    plt.plot(self.bins_center[self.key_pdf] ,self.dir_pdf[self.key_pdf][key_particle], label=key_particle, color=v_colors[i], linewidth=1)
                plt.xlabel("Time [ ns ]")
                plt.legend()
            plt.show()


    def TurnPDFIntoFunction(self):
        self.dir_pdfs_function = {}
        for key in self.dir_pdf[self.key_pdf].keys():
            # z = np.polyfit(self.bins_center_before_cut[self.key_pdf], self.dir_pdf_before_cut[self.key_pdf][key], 30)
            # self.dir_pdfs_function[key] = np.poly1d(z)
            self.dir_pdfs_function[key] = interpolate.interp1d(self.bins_center_before_cut[self.key_pdf], self.dir_pdf_before_cut[self.key_pdf][key],kind="linear")
        if check_poly_fit_pdf:
            x_function = np.linspace(self.bins_center_before_cut[self.key_pdf][0], self.bins_center_before_cut[self.key_pdf][-1], 1000)
            for i,key in enumerate(self.dir_pdf[self.key_pdf].keys()):
                plt.figure()
                plt.scatter(self.bins_center_before_cut[self.key_pdf], self.dir_pdf_before_cut[self.key_pdf][key],color="orange", label="Raw Histogram",marker="+")
                plt.plot( x_function, self.dir_pdfs_function[key](x_function),color="blue", label="Function Values" )
                plt.legend()
                plt.xlabel("Time [ ns ]")
            plt.show()


    def GetGammaRatio(self):
        v_pdg = self.evts_to_fit["pdg_pdgdep"][0] # All elements in pdg_pdgdep are the same so we just get the first one to get the index
        index_lepton = (v_pdg==22) | (v_pdg==11) | (v_pdg==-11)
        index_hadron = [not item for item in index_lepton]
        v_ratio = []
        for i in range(len(self.evts_to_fit["equen_pdgdep"])):
            # v_ratio.append(np.sum(self.evts_to_fit["equen_pdgdep"][i][index_lepton])/np.sum(self.evts_to_fit["equen_pdgdep"][i]) )
            v_ratio.append(
            1-np.sum(self.evts_to_fit["equen_pdgdep"][i][index_hadron]) / self.evts_to_fit["equen"][i])
        v_ratio = np.array(v_ratio)
        self.evts_to_fit["ratio_lepton"] = v_ratio


    def SetTimeProfileToFit(self, i_index:int=0):
        if fitting_with_charge:
            self.h_to_fit = self.evts_to_fit["h_time_with_charge_norm"][i_index]
            self.statistical_errors_to_fit = self.evts_to_fit["statistical_error_time_with_charge"][i_index]
        else:
            self.h_to_fit = self.evts_to_fit["h_time_norm"][i_index]
            self.statistical_errors_to_fit = self.evts_to_fit["statistical_error_time"][i_index]
        self.gamma_ratio = self.evts_to_fit["ratio_lepton"][i_index]
        self.pdg_to_fit = self.evts_to_fit["pdg"][i_index]
        self.limit_time = self.bins_center[self.key_pdf][0]-self.bins_center_before_cut[self.key_pdf][0]

    def SetNInitialValuesToFit(self):
        v_amplitude = np.random.uniform(0, self.upper_limit_amplitude, size=self.n_amplitude_fit_parameters)
        # v_amplitude = np.array([1., 0])
        # v_amplitude = np.ones(self.n_amplitude_fit_parameters)
        if fix_time_parameters:
            v_time = np.zeros(self.n_time_parameters)
        else:
            v_time = np.random.uniform(-self.limit_time, self.limit_time, size=self.n_time_parameters)
        return np.concatenate((v_amplitude, v_time))

    def FitTimeProfile_BestFit(self, v_n_initial):
        v_limit = []
        v_fix = []
        for i in range(self.n_amplitude_fit_parameters):
            v_limit.append((0, self.upper_limit_amplitude))
            v_fix.append(False)
        for i in range(self.n_time_parameters):
            v_limit.append((-(self.bins_center[self.key_pdf][0]-self.bins_center_before_cut[self.key_pdf][0]),
                            (self.bins_center[self.key_pdf][0]-self.bins_center_before_cut[self.key_pdf][0])))
            if fix_time_parameters:
                v_fix.append(True)
            else:
                v_fix.append(False)

        v_error = np.ones(len(v_limit))*0.005

        if least_square_method:
            m = Minuit.from_array_func(self.Chi2Func, v_n_initial,error=v_error, limit=v_limit, errordef=1,fix=v_fix) #iminuit  1.5.4    pypi_0    pypi
        else:
            m = Minuit.from_array_func(self.LikelihoodFunc, v_n_initial,error=v_error, limit=v_limit, errordef=0.5,fix=v_fix) #iminuit  1.5.4    pypi_0    pypi
        m.migrad()

        return m

    def FitTimeProfile_WorkFlow(self, i_index_data_to_fit:int=0):
        self.SetTimeProfileToFit(i_index=i_index_data_to_fit)
        self.v_n_initial = self.SetNInitialValuesToFit()
        m = self.FitTimeProfile_BestFit(v_n_initial=self.v_n_initial)
        num_iterations=0
        while (not (m.fmin['is_valid'] and m.fmin['has_accurate_covar'])):
            if num_iterations > 9:
                break
            print(f"Refitting!! {num_iterations} times")
            num_iterations += 1
            self.v_n_initial = self.SetNInitialValuesToFit()
            m = self.FitTimeProfile_BestFit(v_n_initial=self.v_n_initial)
        return m

    def AppendGammaFitParameter(self, m):
        if constrain_fitting_amplitude_sum_as_1:
            if self.index_gamma == self.n_amplitude_fit_parameters:
                self.v_ratio_gamma_fit.append(1-np.sum(m.np_values()[:self.n_amplitude_fit_parameters]))
            elif self.index_gamma < self.n_amplitude_fit_parameters:
                self.v_ratio_gamma_fit.append(m.np_values()[self.index_gamma])
            else:
                print("ERROR:\tThe index to get parameter of gamma ratio is out of range!!!!")
                exit(1)
        else:
            self.v_ratio_gamma_fit.append(m.np_values()[self.index_gamma]/np.sum(m.np_values()[:self.n_amplitude_fit_parameters]))


    def GetTimeProfileFitResult(self, v_n:np.ndarray):
        v_time_parameters = v_n[self.n_amplitude_fit_parameters:]
        self.fit_result_time_profile = self.GetHistExpected(v_n)
        for i, key in enumerate(self.dir_pdf[self.key_pdf].keys()):
            if constrain_fitting_amplitude_sum_as_1:
                if i == self.n_amplitude_fit_parameters:
                    self.fit_profile_separate[key] = (1-np.sum(v_n[:self.n_amplitude_fit_parameters]))* self.dir_pdfs_function[key](self.bins_center[self.key_pdf]+v_time_parameters[i])
                else:
                    self.fit_profile_separate[key] = v_n[i]*self.dir_pdfs_function[key](self.bins_center[self.key_pdf]+v_time_parameters[i])
            else:
                self.fit_profile_separate[key] = v_n[i] * self.dir_pdfs_function[key](
                    self.bins_center[self.key_pdf] + v_time_parameters[i])

    def PlotPDFTimeProfile(self):
        for key_whether_charge, dir in self.dir_pdf.items():
            plt.figure()
            for key_particle, h_pdf in dir.items():
                plt.plot(self.bins_center[key_whether_charge], h_pdf, label=key_particle)
            plt.title(key_whether_charge)
            plt.xlabel("Time [ ns ]")
            plt.legend()
        plt.show()

    def PlotTimeToFit(self):
        plt.figure()
        for i in range(10):
            plt.plot(self.bins_center[self.key_no_charge], self.evts_to_fit["h_time_norm"][i])
        plt.title("Time Histograms to Fit")
        plt.xlabel("Time [ ns ]")
        plt.show()

    def PlotFitResults(self, m):
        print("Fit Keys:\t", self.dir_pdf[self.key_pdf].keys())
        print("Fit Results:\t", m.np_values())
        print("chi2/ndf:\t", m.fval**2 / (len(self.h_to_fit) - len(self.v_n_initial)))
        print("pdg:\t", self.pdg_to_fit)
        plt.figure()
        self.GetTimeProfileFitResult(v_n=m.np_values())
        PLotDataWithErrorBar_numpy(h_data=self.h_to_fit, h_edges=self.bins[self.key_no_charge], h_y_errors=self.statistical_errors_to_fit, label="Data To Fit")
        plt.plot(self.bins_center[self.key_no_charge],self.fit_result_time_profile, label="Fit Results", linewidth=2, color="Red")
        for i, key in enumerate(self.fit_profile_separate.keys()):
            if constrain_fitting_amplitude_sum_as_1:
                if i == self.n_amplitude_fit_parameters:
                    plt.plot(self.bins_center[self.key_no_charge], self.fit_profile_separate[key],
                             label=f"Fit Results({key}={(1-np.sum(m.np_values()[:self.n_amplitude_fit_parameters])):.4f})", ls="--", linewidth=2)
                else:
                    plt.plot(self.bins_center[self.key_no_charge],self.fit_profile_separate[key], label=f"Fit Results({key}={m.np_values()[i]:.4f})", ls="--", linewidth=2)
            else:
                plt.plot(self.bins_center[self.key_no_charge], self.fit_profile_separate[key],
                         label=f"Fit Results({key}={m.np_values()[i]/np.sum(m.np_values()[:self.n_amplitude_fit_parameters]):.4f})", ls="--", linewidth=2)

        # plt.plot(self.bins_center[self.key_no_charge],self.h_to_fit, label="Profile To Fit")
        if check_result_plot_pdf:
            for key in self.dir_pdfs_function.keys():
                plt.plot(self.bins_center[self.key_pdf], self.dir_pdfs_function[key](self.bins_center[self.key_pdf]), label=f"PDF({key})", ls="-.", linewidth=1)
        plt.xlabel("Time [ ns ]")
        plt.title("$R_{Lepton}=$"+"{:.2f}".format(self.gamma_ratio))
        plt.legend()
        
        if check_whether_bestfit:
            index_non_zero_error = (self.statistical_errors_to_fit != 0.)
            if add_error_in_chi2_fitting:
                v_delta_chi2_fit = np.absolute(self.h_to_fit[index_non_zero_error]-self.fit_result_time_profile[index_non_zero_error])/self.statistical_errors_to_fit[index_non_zero_error]
                v_delta_chi2_pure_gamma = np.absolute(self.h_to_fit[index_non_zero_error]-self.dir_pdfs_function["gamma"](self.bins_center[self.key_pdf])[index_non_zero_error])/self.statistical_errors_to_fit[index_non_zero_error]
                v_delta_chi2_fit = np.nan_to_num(v_delta_chi2_fit)
                v_delta_chi2_pure_gamma = np.nan_to_num(v_delta_chi2_pure_gamma)
            else:
                v_delta_chi2_fit = np.absolute(self.h_to_fit-self.fit_result_time_profile)
                v_delta_chi2_pure_gamma = np.absolute(self.h_to_fit-self.dir_pdfs_function["gamma"](self.bins_center[self.key_pdf]))

            print("Delta Chi2 ( Fit ):\t",np.sum(v_delta_chi2_fit ))
            print("Delta Chi2 ( Pure Gamma PDF ):]\t", np.sum(v_delta_chi2_pure_gamma))
            plt.figure()
            if add_error_in_chi2_fitting:
                plt.plot(self.bins_center[self.key_pdf][index_non_zero_error], v_delta_chi2_fit, label="$\Delta \chi ^2$ ( Fit )")
                plt.plot(self.bins_center[self.key_pdf][index_non_zero_error], v_delta_chi2_pure_gamma, label="$\Delta \chi ^2$ ( Pure Gamma )")
            else:
                plt.plot(self.bins_center[self.key_pdf], v_delta_chi2_fit, label="$\Delta \chi ^2$ ( Fit )")
                plt.plot(self.bins_center[self.key_pdf], v_delta_chi2_pure_gamma, label="$\Delta \chi ^2$ ( Pure Gamma )")

            plt.plot(self.bins_center[self.key_pdf], self.h_to_fit, label="profile to fit")
            plt.plot(self.bins_center[self.key_pdf], self.dir_pdfs_function["gamma"](self.bins_center[self.key_pdf]), label="gamma PDF" )
            plt.plot(self.bins_center[self.key_pdf], self.fit_result_time_profile, label="Fit Result")
            plt.legend()
            plt.xlabel("$\Delta \chi ^2$")

        plt.show()

    def PlotGammaRatioAndFit(self, n_to_fit=5, name_file_to_save:str="fir_result.npz"):
        print(self.evts_to_fit["ratio_lepton"].shape)
        v_lepton_ratio = self.evts_to_fit["ratio_lepton"][:n_to_fit]
        print(v_lepton_ratio)
        print(self.v_ratio_gamma_fit)
        if save_fit_result:
            np.savez(name_file_to_save, gamma_ratio_fit=self.v_ratio_gamma_fit, gamma_ratio_truth=v_lepton_ratio)
        plt.hist2d(self.v_ratio_gamma_fit, v_lepton_ratio, bins=(np.linspace(0, 1, 20), np.linspace(0, 1, 20)), norm=LogNorm())
        plt.colorbar()
        plt.xlabel("Gamma Ratio ( Fit )")
        plt.ylabel("Gamma Ratio ( Truth )")
        plt.title(f"{fitting_source} Samples")
        plt.show()

    def FindBestFitInMultiFitting(self, v_m):
        m_best = v_m[0]
        for i, m in enumerate(v_m[1:]):
            # print(i, m.fval)
            if m.fval < m_best.fval:
                m_best = m
        return m_best

    def UniformLeptonRatioDistribution(self):
        self.evts_to_fit["ratio_lepton"][self.evts_to_fit["ratio_lepton"]<0] = 0.
        h_lepton_ratio, h_edges = np.histogram(self.evts_to_fit["ratio_lepton"], bins=50)
        total_n_samples_to_select = h_lepton_ratio[0]
        n_samples_to_select = np.mean(h_lepton_ratio[1:])
        ratio_abort = 1 - n_samples_to_select/total_n_samples_to_select
        index_need_to_select = np.where(self.evts_to_fit["ratio_lepton"]<=h_edges[1])[0]
        index_abort = random.sample(list(index_need_to_select), int(len(index_need_to_select)*ratio_abort))
        index_remain = np.array([True]*len(self.evts_to_fit["ratio_lepton"]))
        index_remain[index_abort] = False
        for key in self.evts_to_fit.keys():
            if key == "edep":
                continue
            self.evts_to_fit[key] = np.array(self.evts_to_fit[key])[index_remain]

    def ApplyEquenCut(self, Equen_uplimit:float, Equen_downlimit:float):
        index_equen_cut = (Equen_downlimit<self.evts_to_fit["equen"]) & (self.evts_to_fit["equen"]<Equen_uplimit)
        for key in self.evts_to_fit.keys():
            try:
                self.evts_to_fit[key] = self.evts_to_fit[key][index_equen_cut]
            except:
                continue
        plt.figure()
        print(self.evts_to_fit["equen"])
        plt.hist(self.evts_to_fit["equen"])
        plt.show()

    def SplitSamples(self, ratio_train:float=0.7):
        self.total_length_evts = len(self.evts_to_fit[list(self.evts_to_fit.keys())[0]])
        for key in self.evts_to_fit.keys():
            self.evts_to_fit_train[key] = self.evts_to_fit[key][:int(ratio_train*self.total_length_evts)]
            self.evts_to_fit_test[key] = self.evts_to_fit[key][int(ratio_train*self.total_length_evts):]

    def GetVertexR(self):
        self.evts_to_fit["R"] = np.sum((self.evts_to_fit["vertex"]/1000)**2, axis=1)**0.5

    def PrepareDataloader(self):
        if self.training_combine:
            if self.training_add_vertex:
                if self.training_add_energy:
                    self.trainset = CNNDataset(np.concatenate((self.evts_to_fit_train["h_time_norm"],self.evts_to_fit_train["h_time_with_charge_norm"],
                                                               self.evts_to_fit_train["vertex"]/17.5/1000, self.evts_to_fit_train["equen"].reshape(-1, 1)/100), axis=1),
                                                            self.evts_to_fit_train["ratio_lepton"])
                    self.testset = CNNDataset(np.concatenate((self.evts_to_fit_test["h_time_norm"],self.evts_to_fit_test["h_time_with_charge_norm"],
                                                              self.evts_to_fit_test["vertex"]/17.5/1000, self.evts_to_fit_test["equen"].reshape(-1,1)/100),axis=1),
                                                                self.evts_to_fit_test["ratio_lepton"])
                else:
                    self.trainset = CNNDataset(np.concatenate(
                        (self.evts_to_fit_train["h_time_norm"], self.evts_to_fit_train["h_time_with_charge_norm"],
                         self.evts_to_fit_train["vertex"] / 17.5 / 1000), axis=1),
                        self.evts_to_fit_train["ratio_lepton"])
                    self.testset = CNNDataset(np.concatenate(
                        (self.evts_to_fit_test["h_time_norm"], self.evts_to_fit_test["h_time_with_charge_norm"],
                         self.evts_to_fit_test["vertex"] / 17.5 / 1000), axis=1),
                        self.evts_to_fit_test["ratio_lepton"])
                # self.trainset = CNNDataset(np.concatenate((self.evts_to_fit_train["h_time_norm"],self.evts_to_fit_train["h_time_with_charge_norm"],
                #                                            self.evts_to_fit_train["R"].reshape(len(self.evts_to_fit_train['R']),1)/17.5/1000), axis=1),self.evts_to_fit_train["ratio_lepton"])
                # self.testset = CNNDataset(np.concatenate((self.evts_to_fit_test["h_time_norm"],self.evts_to_fit_test["h_time_with_charge_norm"],
                #                                           self.evts_to_fit_test["R"].reshape(len(self.evts_to_fit_test['R']),1)/17.5/1000),axis=1),self.evts_to_fit_test["ratio_lepton"])
            else:
                self.trainset = CNNDataset(np.concatenate((self.evts_to_fit_train["h_time_norm"],self.evts_to_fit_train["h_time_with_charge_norm"]), axis=1),
                                           self.evts_to_fit_train["ratio_lepton"])
                self.testset = CNNDataset(np.concatenate((self.evts_to_fit_test["h_time_norm"],self.evts_to_fit_test["h_time_with_charge_norm"] ),axis=1),
                                          self.evts_to_fit_test["ratio_lepton"])
        else:
            if self.training_with_charge:
                self.trainset = CNNDataset(self.evts_to_fit_train["h_time_with_charge_norm"], self.evts_to_fit_train["ratio_lepton"])
                self.testset = CNNDataset(self.evts_to_fit_test["h_time_with_charge_norm"], self.evts_to_fit_test["ratio_lepton"])
            else:
                self.trainset = CNNDataset(self.evts_to_fit_train["h_time_norm"], self.evts_to_fit_train["ratio_lepton"])
                self.testset = CNNDataset(self.evts_to_fit_test["h_time_norm"], self.evts_to_fit_test["ratio_lepton"])
        self.trainloader = DataLoader(self.trainset, batch_size=30, shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=10, shuffle=False)
        self.evts_to_fit_train.clear()
        # self.evts_to_fit_test.clear()

    def SetModel(self, model_mode, validate_mode=False):
        if not validate_mode:
            self.length_input_train = len(self.trainset[0][0][0])
        else:
            self.length_input_train = len(self.validateset[0][0][0])
        if model_mode == 0:
            self.model = CNN1D()
        elif model_mode == 1:
            self.model = CNN1D_2()
        elif model_mode == 2:
            self.model = Net()
        elif model_mode == 3:
            self.model = LinearNet(self.length_input_train)

        self.criterion = nn.MSELoss()
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00002)


    def TrainNN(self):
        self.PrepareDataloader()
        self.SetModel(model_mode=self.model_mode)
        epochs = 15
        running_loss=0
        self.path_model = f"./model_CNN1D{self.model_label}/"
        self.v_train_loss = []
        self.v_test_loss = []
        print_net_structure = True
        length_train = len(self.trainloader)
        for epoch in range(epochs):
            for time_profile, ratio_lepton in self.trainloader:
                if print_net_structure:
                    self.VisulizeNetwork(input_size=time_profile.shape)
                    print_net_structure = False
                time_profile, ratio_lepton = time_profile.to(device), ratio_lepton.to(device)
                #-----------------Forward Pass----------------------
                output=self.model(time_profile).view((-1))
                loss=self.criterion(output,ratio_lepton)
                # loss=LossFunc(output, ratio_lepton)
                #-----------------Backward Pass---------------------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss+=loss.item()
                # print("########################")
                # print(f"predict:\t", output)
                # print(f"truth:\t", ratio_lepton.detach().cpu().numpy())

            # -----------------Log-------------------------------
            self.v_train_loss.append(running_loss/length_train)
            print("======> epoch: {}/{}, Loss:{}".format(epoch,epochs,self.v_train_loss[-1]))
            running_loss = 0
            self.TestNN()

        self.SaveModel()

        if self.save_ML_output:
            self.SaveOutput()

    def SaveModel(self):
        if not os.path.isdir(self.path_model):
            os.mkdir(self.path_model)
        state = {"net": self.model.state_dict()}
        torch.save(state, self.path_model+"model_0.t7")

    def TestNN(self, validate_mode=False):
        self.model.eval()
        v_output_predict = []
        v_truth = []
        if validate_mode:
            data_loader = self.validateset
        else:
            data_loader = self.testset
        with torch.no_grad():
            running_loss = 0
            for time_profile, ratio_lepton in data_loader:
                time_profile, ratio_lepton = time_profile.to(device), ratio_lepton.to(device)
                output = self.model(time_profile).view((-1))
                loss=self.criterion(output,ratio_lepton)
                # loss=LossFunc(output, ratio_lepton)
                running_loss += loss.item()
                output=output.detach().cpu().numpy()
                ratio_lepton = ratio_lepton.detach().cpu().numpy()
                # print("########################")
                # print(f"predict:\t", output)
                # print(f"truth:\t", ratio_lepton)
                # print(f"truth:\t", type(float(ratio_lepton)))


                v_output_predict+=list(output)
                if ratio_lepton.shape == ():
                    v_truth.append(float(ratio_lepton))
                else:
                    v_truth+=list(ratio_lepton)
            self.v_test_loss.append(running_loss/len(self.testloader))
            print("Test Loss:\t", self.v_test_loss[-1])
        if not validate_mode:
            self.v_output_predict = v_output_predict
            self.v_truth = v_truth
        return (v_output_predict, v_truth)

    def SaveOutput(self):
        self.dir_save = self.path_model
        if not os.path.isdir(self.dir_save):
            os.mkdir(self.dir_save)
        np.savez(self.dir_save+"predict.npz", dir_events=self.evts_to_fit_test, predict=np.array(self.v_output_predict),
                 truth=np.array(self.v_truth), train_loss=self.v_train_loss, test_loss=self.v_test_loss)

    def SetValidateDataset(self, dir_evts_validate:dict):
        self.evts_to_fit_validate = dir_evts_validate
        self.SetValidateDataLoader()

    def SetPredictMode(self, name_file_mode:str):
        if "combine" in name_file_mode:
            self.training_combine = True
        if "add_vertex" in name_file_mode:
            self.training_add_vertex = True
        if "add_energy" in name_file_mode:
            self.training_add_energy = True

    def SetValidateDataLoader(self):
        self.SetBins()
        self.option_h_time = ""
        self.evts_to_fit_validate["h_time_norm"] = []
        self.evts_to_fit_validate["h_time_with_charge_norm"] = []
        for i in range(len(self.evts_to_fit_validate["h_time"])):
            self.evts_to_fit_validate["h_time_norm"].append(
                NormProfile(h_time=self.evts_to_fit_validate["h_time" + self.option_h_time][i][self.index_h_time],
                            h_edges=self.bins[self.key_no_charge], divide_bin_width=self.divide_bin_width))
            self.evts_to_fit_validate["h_time_with_charge_norm"].append(NormProfile(
                    h_time=self.evts_to_fit_validate["h_time_with_charge" + self.option_h_time][i][self.index_h_time_with_charge],
                    h_edges=self.bins[self.key_with_charge], divide_bin_width=self.divide_bin_width))

        if self.training_combine:
            if self.training_add_vertex:
                if self.training_add_energy:
                    self.validateset = CNNDataset(np.concatenate((self.evts_to_fit_validate["h_time_norm"],self.evts_to_fit_validate["h_time_with_charge_norm"],
                                                               self.evts_to_fit_validate["vertex"]/17.5/1000, self.evts_to_fit_validate["equen"].reshape(-1, 1)/100), axis=1),
                                                            self.evts_to_fit_validate["ratio_lepton"])
                else:
                    self.validateset = CNNDataset(np.concatenate(
                        (self.evts_to_fit_validate["h_time_norm"], self.evts_to_fit_validate["h_time_with_charge_norm"],
                         self.evts_to_fit_validate["vertex"] / 17.5 / 1000), axis=1),
                        self.evts_to_fit_validate["ratio_lepton"])
            else:
                self.validateset = CNNDataset(np.concatenate((self.evts_to_fit_validate["h_time_norm"],self.evts_to_fit_validate["h_time_with_charge_norm"]), axis=1),
                                           self.evts_to_fit_validate["ratio_lepton"])
        else:
            if self.training_with_charge:
                self.validateset = CNNDataset(self.evts_to_fit_validate["h_time_with_charge_norm"], self.evts_to_fit_validate["ratio_lepton"])
            else:
                self.validateset = CNNDataset(self.evts_to_fit_validate["h_time_norm"], self.evts_to_fit_validate["ratio_lepton"])
        self.validateloader = DataLoader(self.validateset, batch_size=30, shuffle=True)

    def SetBins(self):
        self.bins_before_cut[self.key_no_charge] = np.array(
            [-19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
             7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
             48, 50, 52, 54, 56, 58, 60, 62, 66, 72, 80, 90, 102, 116, 132, 150, 170, 192, 216, 242, 270, 300, 332, 366,
             402, 440, 480, 522, 566, 612, 660, 710, 762, 816])
        self.bins_before_cut[self.key_with_charge] = self.bins_before_cut[self.key_no_charge]
        self.index_h_time = GetAdjustBinsIndex(GetBinCenter(self.bins_before_cut[self.key_no_charge]), self.bin_start, self.bin_end)
        self.index_h_time_with_charge = GetAdjustBinsIndex(GetBinCenter(self.bins_before_cut[self.key_with_charge]), self.bin_start,
                                                       self.bin_end)
        self.bins[self.key_no_charge] = self.bins_before_cut[self.key_no_charge][1:][self.index_h_time]
        self.bins[self.key_with_charge] = self.bins_before_cut[self.key_with_charge][1:][self.index_h_time_with_charge]

    def LoadModelAndPredict(self, name_file_model:str, model_mode=3):
        self.SetPredictMode(name_file_model)
        self.SetModel(model_mode, validate_mode=True)
        self.model = self.model.to(device)
        if device == "cuda":
            checkpoint = torch.load(name_file_model)
        else:
            checkpoint = torch.load( name_file_model, map_location=torch.device("cpu") )
        self.model.load_state_dict(checkpoint["net"])
        return self.TestNN(validate_mode=True)

    def VisulizeNetwork(self, input_size):
        from torchsummary import summary
        print("===> Input Size:\t", input_size)
        summary(self.model, input_size=input_size)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DSNB component fitter')
    parser.add_argument("--model", "-m", type=int, help="which ML model to use", default=3 )
    args = parser.parse_args()

    plot_pdf = False
    check_poly_fit_pdf = False
    plot_time_to_fit = False
    debug_short_file_list = True
    plot_data_loaded = False
    center_fitting = False
    fitting_with_charge = True
    name_center = "0_0_0"
    use_truth_time = False
    save_fit_result = True
    n_to_fit = -1
    least_square_method = True
    add_error_in_chi2_fitting = False
    use_relative_error = False
    constrain_fitting_amplitude_sum_as_1 = False
    fix_time_parameters = True

    use_ML_method = True

    check_result_plot_fit = False
    check_result_plot_pdf = True
    check_whether_bestfit = False

    n_times_try_diff_initial_value = 10
    # fitting_source = "gamma"
    fitting_source = "neutron"

    if use_ML_method:
        path_data = "/afs/ihep.ac.cn/users/l/luoxj/gpu_500G/DSNB_component_fitting/pdf_time_profile/"
    else:
        path_data = "/afs/ihep.ac.cn/users/l/luoxj/DSNB_component_fitting/code_fitting/pdf_time_profile/"
    # dir_name_files = {"gamma":path_data+"gamma_time_profile.npz",
    #                   "alpha":path_data+"alpha_time_profile.npz",
    #                   "proton":path_data+"proton_time_profile.npz"}
    if center_fitting:
        if use_truth_time:
            dir_name_files = {"gamma":path_data+f"gamma_{name_center}_Truth__time_profile.root",
                          "proton":path_data+f"proton_{name_center}_Truth__time_profile.root"}
        else:
            dir_name_files = {"gamma":path_data+f"gamma_{name_center}_time_profile.root",
                              "proton":path_data+f"proton_{name_center}_time_profile.root"}
    else:
        dir_name_files = {"gamma":path_data+"gamma_time_profile.root",
                      "alpha":path_data+"alpha_time_profile.root",
                      "proton":path_data+"proton_time_profile.root"}
    FLH = FitWithComponent()

    FLH.LoadTimeProfilePDF(dir_name_files=dir_name_files)
    FLH.PlotPDFTimeProfile() if plot_pdf else 0
    FLH.TurnPDFIntoFunction()

    if not center_fitting:
        FLH.GetFileListToFit(name_source="Neutron", short_file_list=debug_short_file_list)
    else:
        FLH.GetFileListToFit(name_source="neutron_0_0_0", short_file_list=debug_short_file_list)
    # FLH.GetFileListToFit(name_source=f"{fitting_source}_0_0_0", short_file_list=debug_short_file_list)

    FLH.LoadDatToFit()
    FLH.GetGammaRatio()
    FLH.PlotTimeToFit() if plot_time_to_fit else 0
    n_to_fit = (FLH.entries if n_to_fit==-1 else n_to_fit)
    if not use_ML_method:
        from PlotErrorBar import PLotDataWithErrorBar_numpy
        for i in tqdm.trange(n_to_fit):
            if i > FLH.entries:
                break
            v_m = []
            for j in range(n_times_try_diff_initial_value):
                v_m.append(FLH.FitTimeProfile_WorkFlow(i_index_data_to_fit=i))
            m_best = FLH.FindBestFitInMultiFitting(v_m)
            FLH.AppendGammaFitParameter(m_best)
            if check_result_plot_fit:
                FLH.PlotFitResults(m_best)
        FLH.PlotGammaRatioAndFit(n_to_fit=n_to_fit, name_file_to_save=f"./{fitting_source}_fitting_result.npz")
        plt.show()
    else:
        if not FLH.load_uniform_R_lepton_data:
            FLH.UniformLeptonRatioDistribution()
        if FLH.training_with_equen_cut:
            FLH.ApplyEquenCut(Equen_uplimit=30, Equen_downlimit=10)
        FLH.SplitSamples(ratio_train=0.95)
        FLH.TrainNN()




