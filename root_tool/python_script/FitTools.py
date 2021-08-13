# -*- coding:utf-8 -*-
# @Time: 2021/7/13 16:11
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitTools.py
import matplotlib.pylab as plt
import numpy as np
from array import array
import ROOT
from root_numpy import array2hist
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

def ArrayToROOTHist(h:np.array,edges:np.array , name_h="hist_root"):
    h_root = ROOT.TH1D(name_h, name_h, len(edges)-1, array("d", edges))
    array2hist(h, h_root)
    return h_root


def Fit1DHist(h:np.array,edges:np.array ,name_func_fit:str, name_h="hist_root"):
    h_root = ROOT.TH1D(name_h, name_h, len(edges)-1, array("d", edges))
    array2hist(h, h_root)
    h_root.Fit(name_func_fit)
    return h_root

