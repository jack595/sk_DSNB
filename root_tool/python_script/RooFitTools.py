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
    data_array = np.array(data_array, dtype=[(name_in_tree, dtype)])
    return rn.array2tree(data_array)


def DictToRNArray(dir_events:dict):
    v_dtype = []
    for key,item in dir_events.items():
        if isinstance(item[0], np.ndarray):
            # for an entry is array, we can use np.array([(1, 2.5, [3.4,1.2]),(4, 5, [6.8,2.4])],dtype=[('a', np.int32),('b', np.float32),('c', np.float64, (2,))])
            # to make it through, refer to https://github.com/scikit-hep/root_numpy/issues/376
            v_dtype.append((key, np.dtype(item[0][0]), (len(item[0]),)))
        else:
            v_dtype.append((key, np.dtype(item[0])))
    list_data_save = [item for _, item in dir_events.items()]
    list_data_save = [ tuple(i_entry_data) for i_entry_data in np.array(list_data_save,dtype=object).T]

    return np.array(list_data_save, dtype=v_dtype)

def DictToTree(dir_events:dict, name_tree:str="tree"):
    array_data = DictToRNArray(dir_events)
    tree = rn.array2tree(array_data, tree=name_tree)
    tree.Scan()
    return tree

def SaveDictToTFile(dir_events:dict, name_tree:str, name_files:str):
    array_data = DictToRNArray(dir_events)
    rn.array2root(array_data, treename=name_tree, filename=name_files,mode="recreate")


# Just An Example
def GetGaussianFunc():
    x = ROOT.RooRealVar("Time","Time",0,800)
    mean = ROOT.RooRealVar("mean", "mean", 100, 0, 800)
    sigma = ROOT.RooRealVar("sigma", "sigma", 80, 0.1, 1000)
    gx = ROOT.RooGaussian("gx", "gx", x, mean, sigma)

def TreeToDataset(x, tree):
    return ROOT.RooDataSet("data","data",ROOT.RooArgSet(x), ROOT.RooFit.Import(tree))

