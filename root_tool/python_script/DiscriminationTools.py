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
from HistTools import GetBinCenter

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

class DiscriminationTools:
    def __init__(self, key_0="Background", key_1="Signal"):
        self.dir_events = {}
        self.dir_n_samples = {}
        self.dir_hist_PSD = {}
        self.f_BkgIneff2SigEff = None
        self.dir_eff_to_df = {"index":[],f"{key_1} Eff.":[], f"{key_0} Ineff.":[],
                              rf"({key_1} Residue)/(Total Residue)":[],
                              rf"({key_0} Residue)/(Total Residue)":[],
                              "PSD Cut":[]}

        self.ax_ROC = None
        self.ax_PSD = None
        self.key_0 = key_0
        self.key_1 = key_1
        self.global_PSD_cut = None


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

    def GetPSDDistribution(self, dir_events=None, v_tags=None, bins= np.linspace(0, 1, 100),ax=None,title_options="", *args, **kwargs):
        if v_tags is None:
            v_tags = [self.key_0, self.key_1]
        if ax is None:
            if self.ax_PSD == None:
                plt.figure()
                self.ax_PSD = plt.subplot(111)
            ax = self.ax_PSD
        if dir_events == None:
            dir_events = self.dir_events

        self.bins = bins
        self.bins_center = GetBinCenter(bins)

        for i, key in enumerate([0,1]):
            self.dir_hist_PSD[key] = ax.hist( (dir_events["PSD"][dir_events["evtType"]==key]) ,bins=bins,
                     histtype="step", label=v_tags[i],*args, **kwargs)
        ax.semilogy()
        ax.legend()
        ax.set_xlabel("PSD Output")
        ax.set_title("PSD Distribution"+title_options)

    def PlotROCCurves(self,xlim=None,ylim=None, ax=None, *args, **kargs):
        if ax is None:
            if self.ax_ROC == None:
                plt.figure()
                self.ax_ROC = plt.subplot(111)
            ax = self.ax_ROC

        n0 = self.dir_hist_PSD[0][0]
        n1 = self.dir_hist_PSD[1][0]
        eff_bkg = []
        eff_sig = []
        for i in range(len(n0)):
            eff_bkg.append(np.sum(n0[i:]) * 1.0 / np.sum(n0))
            eff_sig.append(np.sum(n1[i:]) * 1.0 / np.sum(n1))

        # Map background inefficiency to signal efficiency
        self.f_BkgIneff2SigEff = interp1d(eff_bkg, eff_sig)

        # Map background inefficiency to PSD cut
        self.f_BkgIneff2PSDCut = interp1d(eff_bkg, self.bins_center)
        self.f_PSDCut2BkgIneff = interp1d(self.bins_center, eff_bkg)

        ax.plot(eff_bkg,eff_sig, *args, **kargs)
        ax.set_xlabel(f'{self.key_0} Inefficiency')
        ax.set_ylabel(f'{self.key_1} efficiency')
        ax.set_title("Efficiency Curves")

        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])

        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

    def PlotROCCurvesDiffEBins(self, bins_Energy, key_Energy="Erec",option="", AppendAnothorOptionLine=False, label_AppendOption=None,
                               v_colors=None, ls="-", ax_ROC=None, xlim=None, ylim=None, v_bkg_ineff=None):
        if ax_ROC==None:
            plt.figure("ROC")
            ax_ROC = plt.subplot(111)

        for i_bin in range( len(bins_Energy)-1 ):
            if AppendAnothorOptionLine:
                if i_bin == 0:
                    label = label_AppendOption
                else:
                    label = None

                if v_colors != None:
                    color = v_colors[i_bin]
                else:
                    color = None
            else:
                label = f"{bins_Energy[i_bin]:.2f} < "+ "$E_{rec}$"+ f" < {bins_Energy[i_bin+1]:.2f} MeV"
                color = None

            fig_PSD,ax_PSD = plt.subplots(1,1)
            dir_events_in_quench_bins = {}
            index_energy = ( self.dir_events[key_Energy]<bins_Energy[i_bin+1] ) & ( self.dir_events[key_Energy]>bins_Energy[i_bin] )
            for key in self.dir_events.keys():
                dir_events_in_quench_bins[key] = self.dir_events[key][index_energy]

            self.GetPSDDistribution(dir_events_in_quench_bins,
                                    title_options=f" {bins_Energy[i_bin]:.2f} < Erec< {bins_Energy[i_bin+1]:.2f} MeV {option}",
                                    ax=ax_PSD)


            self.PlotROCCurves(ax=ax_ROC, color=color, ls=ls, label=label, xlim=xlim, ylim=ylim)
            self.MaximumSignificance(v_bkg_ineff, ax=ax_ROC, condition=label)
            if self.global_PSD_cut != None:
                self.CertainPSDCut(self.global_PSD_cut, condition=label+"(Global Cut)")
            
        ax_ROC.legend()
        return ax_ROC



    def MaximumSignificance(self,v_bkg_ineff=None,set_global_PSD_cut=False , *args, **kwargs):
        """
        Maximum Significance S/sqrt(S+B) to set PSD cut
        :param v_bkg_ineff: scan range of background inefficiency
        :param ax:
        :param condition: label for summary table
        :return:
        """
        if v_bkg_ineff is None:
            v_bkg_ineff = np.linspace(0.00015, 0.1, 1000)
        v_sig_eff = self.f_BkgIneff2SigEff(v_bkg_ineff)
        v_PSD_cut = self.f_BkgIneff2PSDCut(v_bkg_ineff)

        #TODO: n_total_sig need to adjust with different energy bin!!!
        n_total_sig = self.dir_n_samples["total"][1]
        n_total_bkg = self.dir_n_samples["total"][0]
        significance = n_total_sig*v_sig_eff/np.sqrt( n_total_bkg*v_bkg_ineff + n_total_sig*v_sig_eff )

        index_max = np.argmax(significance)
        self.CalculateEfficiency( v_PSD_cut[index_max], v_bkg_ineff[index_max], v_sig_eff[index_max],option="Optimized ",
                                  *args, **kwargs)

        if set_global_PSD_cut:
            self.global_PSD_cut = v_PSD_cut[index_max]

    def CertainPSDCut(self, PSD_cut, *args, **kwargs):
        bkg_ineff = self.f_PSDCut2BkgIneff(PSD_cut)
        sig_eff = self.f_BkgIneff2SigEff(bkg_ineff)
        self.CalculateEfficiency( PSD_cut, bkg_ineff, sig_eff,option="Certain PSD Cut ",
                                  *args, **kwargs)

    def CalculateEfficiency(self, PSD_cut, bkg_ineff, sig_eff, option="",ax=None,condition:str=""):

        n_total_sig = self.dir_n_samples["total"][1]
        n_total_bkg = self.dir_n_samples["total"][0]

        n_sig = len( self.dir_events["PSD"][self.dir_events["evtType"]==1] )
        n_bkg = len( self.dir_events["PSD"][self.dir_events["evtType"]==0] )

        print(f"PSD Cut: {PSD_cut:.3g}")

        # Calculate Signal Efficiency and Background Inefficiency
        bkg_optimized = f"{bkg_ineff*100:.3f} % +- {np.sqrt( bkg_ineff*(1-bkg_ineff)*n_bkg )/n_bkg *100:.2g}"
        sig_optimized = f"{sig_eff*100:.3f} % +- {np.sqrt( sig_eff*(1-sig_eff)*n_sig )/n_sig *100:.2g}"


        print(f"{option} Efficiency:\n",f"{self.key_0} inefficiency:\t",bkg_optimized,
                  "\n"+f"{self.key_1} efficiency:\t", sig_optimized)

        # Calculate Residual Ratio
        n_total_residue = n_total_sig*sig_eff + n_total_bkg*bkg_ineff
        ratio_sig2residue = n_total_sig*sig_eff/n_total_residue
        ratio_bkg2residue = n_total_bkg*bkg_ineff/n_total_residue

        str_ratio_sig2residue = f"{ratio_sig2residue*100:.3f} % +- {np.sqrt(ratio_sig2residue*(1-ratio_sig2residue)*n_total_residue)/n_total_residue*100:.2g}"
        str_ratio_bkg2residue = f"{ratio_bkg2residue*100:.3f} % +- {np.sqrt(ratio_bkg2residue*(1-ratio_bkg2residue)*n_total_residue)/n_total_residue*100:.2g}"


        print(f"\n{option} Ratio of Residue:\n", f"{self.key_0} Ratio:\t", str_ratio_bkg2residue,
                f"\n{self.key_1} Ratio:\t", str_ratio_sig2residue )
        print("\n##############################################\n")

        if ax is None:
            ax = self.ax_ROC

        ax.scatter(bkg_ineff, sig_eff)

        self.dir_eff_to_df[f"{self.key_1} Eff."].append(sig_optimized)
        self.dir_eff_to_df[f"{self.key_0} Ineff."].append(bkg_optimized)
        self.dir_eff_to_df[f"({self.key_0} Residue)/(Total Residue)"].append(str_ratio_bkg2residue)
        self.dir_eff_to_df[f"({self.key_1} Residue)/(Total Residue)"].append(str_ratio_sig2residue)
        self.dir_eff_to_df["index"].append(condition)
        self.dir_eff_to_df["PSD Cut"].append(f"{PSD_cut:.2g}")

        return sig_optimized, bkg_optimized, str_ratio_sig2residue, str_ratio_bkg2residue

    def SetNSamplesDict(self, dir_n_samples:dict):
        """

        :param dir_n_samples: for example, {'train': Counter({0: 20044, 1: 20044}), 'test': Counter({0: 335451, 1: 30067}), 'total': Counter({0: 355495, 1: 50111})}
        :return:
        """
        self.dir_n_samples = copy(dir_n_samples)

    def Legend(self):
        self.ax_ROC.legend()
        self.ax_PSD.legend()

    def PrintEffDataframe(self):
        """
        Summary Efficiency Table for executed MaximumSignificance()
        :return:
        """

        df_eff = pd.DataFrame.from_dict(self.dir_eff_to_df)
        self.Legend()
        return  df_eff

if __name__ == '__main__':
    discrimination_tool = DiscriminationTools(key_0="pES", key_1="eES")
    discrimination_tool.GetPredictionData("/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/code/eESAndpESDiscrimination/predict_Combine.npz")
    discrimination_tool.GetPSDDistribution()
    discrimination_tool.PlotROCCurves()
    discrimination_tool.MaximumSignificance(v_bkg_ineff=np.linspace(0.01, 0.2,1000), condition="Total Samples")
    discrimination_tool.PlotROCCurvesDiffEBins(bins_Energy=np.linspace(0.2, 3, 10), xlim=(0, 0.1), ylim=(0.5,1),
                                               v_bkg_ineff=np.linspace(0.001, 0.2,1000))
    discrimination_tool.PrintEffDataframe()


