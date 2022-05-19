# -*- coding:utf-8 -*-
# @Time: 2022/5/16 10:07
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitChargeSpectrum.py
# import matplotlib.pylab as plt
import numpy as np

# plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
import ROOT
from RooFitTools import ArrayToTree




if __name__ == "__main__":
    with np.load("Spectrum.npz", allow_pickle=True) as f:
        dict_TQ_diff_source = f["dir_TQ_diff_source"].item()

    for i, (key,dict_TQ) in enumerate(dict_TQ_diff_source.items()):
        v_Q = dict_TQ["Q"][(dict_TQ["valley"]>-2)]

        x_min, x_max = 0, 1000
        x = ROOT.RooRealVar("Charge", "Charge", x_min, x_max)
        mean = ROOT.RooRealVar("mean", "mean", 0, x_min, x_max )
        sigma = ROOT.RooRealVar("sigma", "sigma", 100, 0.1, 500)

        gauss = ROOT.RooGaussian("gauss", "gauss", x, mean, sigma)

        tree_delta_z = ArrayToTree(v_Q,name_in_tree='Charge')
        data_delta_z = ROOT.RooDataSet("data", "data", ROOT.RooArgSet(x), ROOT.RooFit.Import(tree_delta_z))

        x.setRange("signal", -200, 200)

        fit_result = gauss.fitTo(data_delta_z, ROOT.RooFit.Range("signal"), ROOT.RooFit.Save())
        # print("Chi2:\t", fit_result.minNll())

        locals()[f"c{i}"] = ROOT.TCanvas(f"c_time_{i}","")
        xframe = x.frame()
        data_delta_z.plotOn(xframe)
        gauss.plotOn(xframe)
        gauss.paramOn(xframe, ROOT.RooFit.Layout(0.6,0.9,0.9),ROOT.RooFit.ShowConstants(True))
        xframe.Draw()
        locals()[f"c{i}"].Draw()


        break