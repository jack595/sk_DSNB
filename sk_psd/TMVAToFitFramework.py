# -*- coding:utf-8 -*-
# @Time: 2021/1/18 21:55
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: TMVAToFitFramework.py
import numpy as np
import matplotlib.pylab as plt
import uproot3 as up
from collections import Counter
from matplotlib.colors import LogNorm

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
def DataToFitFramework(name_file:str):
    dir_return = {}
    f = up.open(name_file)
    train_tree = f["bkgRejection/TrainTree"]
    dir_return["labels"]= train_tree.array("id_tag")
    dir_return["equen"] = train_tree.array("Eqe_tag")
    dir_return["predict_proba"] = train_tree.array("BDTG")
    dir_return["isoz"] = train_tree.array("isoz")
    dir_return["ison"] = train_tree.array("ison")
    dir_return["vertex"] = np.zeros((len(dir_return["labels"]), 3))
    ChangeTagIntoLabel(dir_return)
    return dir_return
def ChangeTagIntoLabel(dir_return:dict):
    index_bkg = (dir_return["labels"]==2)
    index_sig = (dir_return["labels"]==1)
    dir_return["labels"][index_sig] = 1
    dir_return["labels"][index_bkg] = 0
    return dir_return
def ScanBDTGEfficiency(dir_return:dict, eff_sig_needed:float=0.8):
    predict_0 = dir_return["predict_proba"]
    labels = dir_return["labels"]
    predict_sig = predict_0[labels==1]
    predict_bkg = predict_0[labels==0]
    print("BDGT limit:\t", np.min(predict_0),
                                  np.max(predict_0))
    for BDTG_cut in np.arange(0.89, 1, 0.0001):
        eff_sig = Counter(predict_sig>BDTG_cut)[True]/len(predict_sig)
        eff_bkg = Counter(predict_bkg>BDTG_cut)[True]/len(predict_bkg)
        if abs(eff_sig-eff_sig_needed) <0.0005:
            print(f"################BDTG cut: {BDTG_cut}#####################")
            print("Signal Eff.:\t", eff_sig)
            print("Background Eff.:\t", eff_bkg)
            print("##########################################################")
            print(f"BDTF cut:\t {BDTG_cut}")
            return (BDTG_cut, eff_sig, eff_bkg)
def GetIndexC11(dir_return:dict):
    v_isoz = dir_return["isoz"] # number of protons
    v_ison = dir_return["ison"]
    index_C11 = (v_isoz==6) & (v_ison==11)
    index_no_C11 = ([not elem for elem in index_C11]) & (dir_return["labels"]==0)
    print("n of NC bkg:\t", Counter(dir_return["labels"]==0)[True])
    print("n of no C11:\t", Counter(index_no_C11)[True])
    print("n of C11:\t", Counter(index_C11)[True])
    return index_C11, index_no_C11
def GetC11Events(dir_return:dict, index_C11:np.ndarray):
    dir_C11 = {}
    for key in dir_return.keys():
        dir_C11[key] = dir_return[key][index_C11]
    return dir_C11
def SaveDict(name_file:str, dir_save:dict):
    np.savez(name_file, labels=dir_save["labels"],
             equen=dir_save["equen"],
             predict_proba=dir_save["predict_proba"],
             vertex=dir_save["vertex"],ison=dir_save["ison"],
             isoz=dir_save["isoz"])

if __name__ == "__main__":
    dir_TMVA_model = "/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_TMVA/"
    dir_save_within16m = DataToFitFramework(dir_TMVA_model+"tmvawr3_wcharge_v0.root")
    dir_save_outside16m = DataToFitFramework(dir_TMVA_model+"tmva_wcharge_v5.root")
    
    print("outside 16m:\t", dir_save_outside16m)
    print("within 16m:\t", dir_save_within16m)
    index_C11_within16m, index_no_C11_within16m = GetIndexC11(dir_save_within16m)
    index_C11_outside16m, index_no_C11_outside16m = GetIndexC11(dir_save_outside16m)
    dir_C11_within16m = GetC11Events(dir_save_within16m, index_C11_within16m)
    dir_C11_outside16m = GetC11Events(dir_save_outside16m, index_C11_outside16m)

    (BDTG_cut_within16m, eff_sig_within16m, eff_bkg_within16m) = ScanBDTGEfficiency(dir_save_within16m, eff_sig_needed=0.8)
    (BDTG_cut_outside16m,eff_sig_outside16m,eff_bkg_outside16m) = ScanBDTGEfficiency(dir_save_outside16m, eff_sig_needed=0.7)
    np.savez(dir_TMVA_model+"predict_0_within16m.npz", labels=dir_save_within16m["labels"],
             equen=dir_save_within16m["equen"],
             predict_proba=dir_save_within16m["predict_proba"],
             vertex=dir_save_within16m["vertex"],ison=dir_save_within16m["ison"],
             isoz=dir_save_within16m["isoz"],index_C11=index_C11_within16m,index_no_C11=index_no_C11_within16m,
             BDTG_cut=BDTG_cut_within16m, eff_sig=eff_sig_within16m, eff_bkg=eff_bkg_within16m)
    np.savez(dir_TMVA_model+"predict_0_outside16m.npz", labels=dir_save_outside16m["labels"],
             equen=dir_save_outside16m["equen"],
             predict_proba=dir_save_outside16m["predict_proba"],
             vertex=dir_save_outside16m["vertex"],index_C11=index_C11_outside16m,index_no_C11=index_no_C11_outside16m,
             ison=dir_save_outside16m["ison"], isoz=dir_save_outside16m["isoz"],
             BDTG_cut=BDTG_cut_outside16m, eff_sig=eff_sig_within16m, eff_bkg=eff_bkg_outside16m)
    np.savez(dir_TMVA_model+"predict_0_outside16m_C11.npz", dir_events=dir_C11_outside16m)
    np.savez(dir_TMVA_model+"predict_0_within16m_C11.npz", dir_events=dir_C11_within16m)
    # SaveDict(dir_TMVA_model+"predict_0_outside16m_C11.npz", dir_C11_outside16m)
    # SaveDict(dir_TMVA_model+"predict_0_within16m_C11.npz", dir_C11_within16m)

