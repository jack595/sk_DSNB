# -*- coding:utf-8 -*-
# @Time: 2021/11/7 21:59
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: RooFitTools.py
import matplotlib.pylab as plt
import numpy as np
import ROOT
import root_numpy as rn

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def ArrayToTree(data_array, name_in_tree, dtype=np.float32):
    data_array = np.array(data_array, dtype=(name_in_tree, dtype))
    return rn.array2tree(data_array)

# Just An Example
def GetGaussianFunc():
    x = ROOT.RooRealVar("Time","Time",0,800)
    mean = ROOT.RooRealVar("mean", "mean", 100, 0, 800)
    sigma = ROOT.RooRealVar("sigma", "sigma", 80, 0.1, 1000)
    gx = ROOT.RooGaussian("gx", "gx", x, mean, sigma)

def TreeToDataset(x, tree):
    return ROOT.RooDataSet("data","data",ROOT.RooArgSet(x), ROOT.RooFit.Import(tree))
