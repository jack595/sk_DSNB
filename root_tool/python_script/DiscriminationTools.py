# -*- coding:utf-8 -*-
# @Time: 2022/3/9 15:20
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: DiscriminationTools.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys
from scipy.interpolate import interp1d
from copy import copy
import pandas as pd
from IPython.display import display

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

class DiscriminationTools:
    def __init__(self):
        self.dir_events = {}
        self.dir_n_samples = {}
        self.dir_hist_PSD = {}
        self.f_BkgIneff2SigEff = None
        self.dir_eff_to_df = {"index":[],r"Signal Eff.":[], r"Background Eff.":[],
                              r"(Signal Residue)/(Total Residue)":[],
                              r"(Background Residue)/(Total Residue)":[]}

        plt.figure()
        self.ax_PSD = plt.subplot(111)
        plt.figure()
        self.ax_ROC = plt.subplot(111)

    def GetPredictionData(self, path_prediction_file:str):
        with np.load( path_prediction_file, allow_pickle=True) as f:
            self.dir_events = f["dir_events"].item()
            print(self.dir_events.keys())
            try:
                self.dir_n_samples = f["dir_n_samples"].item()
                print(self.dir_n_samples)
            except Exception as e:
                print("Cannot get dir_n_samples!!! You need to set it by hand !!Continue")
                pass

    def GetPSDDistribution(self, v_tags=None, bins= np.linspace(0, 1, 100),ax=None,title_options="", *args, **kwargs):
        if v_tags is None:
            v_tags = ["Background", "Signal"]
        if ax is None:
            ax = self.ax_PSD

        for i, key in enumerate([0,1]):
            self.dir_hist_PSD[key] = ax.hist( (self.dir_events["PSD"][self.dir_events["evtType"]==key]) ,bins=bins,
                     histtype="step", label=v_tags[i],*args, **kwargs)
        ax.semilogy()
        ax.legend()
        ax.set_xlabel("PSD Output")
        ax.set_title("PSD Distribution"+title_options)

    def PlotROCCurves(self, xlim=None,ax=None, *args, **kargs):
        if ax is None:
            ax = self.ax_ROC
        n0 = self.dir_hist_PSD[0][0]
        n1 = self.dir_hist_PSD[1][0]
        eff_bkg = []
        eff_sig = []
        for i in range(len(n0)):
            eff_bkg.append(np.sum(n0[i:]) * 1.0 / np.sum(n0))
            eff_sig.append(np.sum(n1[i:]) * 1.0 / np.sum(n1))

        self.f_BkgIneff2SigEff = interp1d(eff_bkg, eff_sig)

        ax.plot(eff_bkg,eff_sig, *args, **kargs)
        ax.set_xlabel('Background Inefficiency')
        ax.set_ylabel('Signal efficiency')
        ax.set_title("Efficiency Curves")

        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])

    def MaximumSignificance(self,v_bkg_ineff=None, ax=None, condition:str=""):
        if v_bkg_ineff is None:
            v_bkg_ineff = np.linspace(0.00015, 0.1, 1000)
        v_sig_eff = self.f_BkgIneff2SigEff(v_bkg_ineff)

        n_total_sig = self.dir_n_samples["total"][1]
        n_total_bkg = self.dir_n_samples["total"][0]
        significance = n_total_sig*v_sig_eff/np.sqrt( n_total_bkg*v_bkg_ineff + n_total_sig*v_sig_eff )

        index_max = np.argmax(significance)
        n_sig = len( self.dir_events["PSD"][self.dir_events["evtType"]==1] )
        n_bkg = len( self.dir_events["PSD"][self.dir_events["evtType"]==0] )

        bkg_optimized = f"{v_bkg_ineff[index_max]*100:.3f} % +- {np.sqrt( v_bkg_ineff[index_max]*(1-v_bkg_ineff[index_max])*n_bkg )/n_bkg *100:.2g}"
        sig_optimized = f"{v_sig_eff[index_max]*100:.3f} % +- {np.sqrt( v_sig_eff[index_max]*(1-v_sig_eff[index_max])*n_sig )/n_sig *100:.2g}"


        print("Optimized Efficiency:\n","Background inefficiency:\t",bkg_optimized,
                  "\nSignal efficiency:\t", sig_optimized)

        n_total_residue = n_total_sig*v_sig_eff[index_max] + n_total_bkg*v_bkg_ineff[index_max]
        ratio_sig2residue = n_total_sig*v_sig_eff[index_max]/n_total_residue
        ratio_bkg2residue = n_total_bkg*v_bkg_ineff[index_max]/n_total_residue

        str_ratio_sig2residue = f"{ratio_sig2residue*100:.3f} % +- {np.sqrt(ratio_sig2residue*(1-ratio_sig2residue)*n_total_residue)/n_total_residue*100:.2g}"
        str_ratio_bkg2residue = f"{ratio_bkg2residue*100:.3f} % +- {np.sqrt(ratio_bkg2residue*(1-ratio_bkg2residue)*n_total_residue)/n_total_residue*100:.2g}"


        print("\nOptimized Ratio of Residue:\n", "Background Ratio:\t", str_ratio_bkg2residue,
                "\nSignal Ratio:\t", str_ratio_sig2residue )
        print("\n##############################################\n")

        if ax is None:
            ax = self.ax_ROC

        ax.scatter(v_bkg_ineff[index_max], v_sig_eff[index_max])

        self.dir_eff_to_df["Signal Eff."].append(sig_optimized)
        self.dir_eff_to_df["Background Eff."].append(bkg_optimized)
        self.dir_eff_to_df["(Background Residue)/(Total Residue)"].append(str_ratio_bkg2residue)
        self.dir_eff_to_df["(Signal Residue)/(Total Residue)"].append(str_ratio_sig2residue)
        self.dir_eff_to_df["index"].append(condition)

        return sig_optimized, bkg_optimized, str_ratio_sig2residue, str_ratio_bkg2residue

    def SetNSamplesDict(self, dir_n_samples:dict):
        self.dir_n_samples = copy(dir_n_samples)

    def Legend(self):
        self.ax_ROC.legend()
        self.ax_PSD.legend()

    def PrintEffDataframe(self):
        df_eff = pd.DataFrame.from_dict(self.dir_eff_to_df)
        self.Legend()
        return  df_eff


