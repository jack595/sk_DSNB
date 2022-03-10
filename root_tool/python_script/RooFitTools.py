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

def MultiArrayToTree(v_data_arrays, v_name_in_tree, dtype=np.float32):
    v_array_to_tree = []
    for i in range(len(v_data_arrays[0])):
        v_array_to_tree.append((v_data_arrays[j][i] for j in range(len(v_data_arrays))))


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

def FitFromArray(v_x,x_range=None,std_get_range=False,x_range_fit=None, sigma_init=1., sigma_range=None,bins=100, func=None, canvas=None, xlabel="x",
                 path_savefig=None, title=None, draw=True, fit_range_sigma=None):
    from HistTools import GetBinCenter
    hist = np.histogram(v_x, bins=bins)
    x_peak = GetBinCenter(hist[1])[np.argmax(hist[0])]
    print(x_peak)
    if fit_range_sigma == None:
        x_std = np.std(v_x)
    else:
        x_std = fit_range_sigma

    if sigma_range == None:
        sigma_range = (0.1, 300)

    if x_range == None:
        if std_get_range:
            x_range = (x_peak-2*x_std, x_peak+2*x_std)
        else:
            x_range = (min(v_x), max(v_x))

    print("----------> Set RooRealVar x")
    x = ROOT.RooRealVar("x", xlabel, x_range[0], x_range[1])

    # Construct signal pdf
    print("-----------> Construct Signal pdf")
    mean = ROOT.RooRealVar("mean", "mean", x_peak, min(v_x), max(v_x))

    sigma = ROOT.RooRealVar("sigma", "sigma", sigma_init, sigma_range[0], sigma_range[1])

    if func == None:
        gx = ROOT.RooGaussian("gx", "gx", x, mean, sigma)
        f = ROOT.RooRealVar("f", "f", 0.5, 0.0, 1.0)
        func = gx
     ###########################################################

    # Generate RooDataset
    print("----------> Generate RooDataset")
    v_x_array = np.array(v_x, dtype=[("x",np.float32)])
    tree_time = rn.array2tree(v_x_array,name="x")
    data = ROOT.RooDataSet("data","data",ROOT.RooArgSet(x), ROOT.RooFit.Import(tree_time))

    # Set Range for x
    print("----------> Set Range for x")
    if x_range_fit == None:
        x_range_fit = (x_peak-x_std, x_peak+x_std)

    x.setRange("signal", x_range_fit[0], x_range_fit[1])

    # Fit data
    fit_result = func.fitTo(data, ROOT.RooFit.Range("signal"), ROOT.RooFit.Save())
    # print("Chi2:\t", fit_result.minNll())

    # Plot Fit Results
    if canvas == None:
        canvas = ROOT.TCanvas("c","")
    canvas.cd()
    xframe = x.frame()
    data.plotOn(xframe)
    func.plotOn(xframe)
    func.paramOn(xframe, ROOT.RooFit.Layout(0.6,0.9,0.9),ROOT.RooFit.ShowConstants(True))
    if title != None:
        xframe.SetTitle(title)
    xframe.Draw()
    if draw:
        canvas.Draw()
    if path_savefig!=None:
        canvas.SaveAs(path_savefig)

    return (mean.getVal(0), sigma.getVal(0))


if __name__ == '__main__':
    import random
    v_data = [random.gauss(0, 1) for _ in range(4000)]
    print(v_data)
    FitFromArray(v_data, xlabel="Time")
    plt.show()



