# -*- coding:utf-8 -*-
# @Time: 2022/4/24 21:55
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotSpectrum.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def PlotSpectrumComponents(df_residue,E_bins=np.linspace(0,100,80),
                           logy=True):
    v2d_Erec = []
    v_labels = []
    v_nEvts = []

    plt.hist(df_residue["recE"],bins=E_bins,histtype="step",
             color="black", linewidth=3)
    for tag_truth in set(df_residue["evtType"]):
        v_Erec =  np.array(df_residue["recE"][(df_residue["evtType"]==tag_truth) ])
        v2d_Erec.append(v_Erec)
        v_labels.append(tag_truth)
        v_nEvts.append(len(v_Erec))

    (v_nEvts, v_labels, v2d_Erec) = list( zip( *sorted( zip(v_nEvts, v_labels,v2d_Erec) ) ) )
    plt.hist( v2d_Erec, bins=E_bins,ls="--", stacked=True, label=v_labels,)
    plt.legend()
    plt.xlabel("$E_{rec}$ [ MeV ]")
    plt.ylabel("N of Events")
    plt.title("Spectrum Components")
    if logy:
        plt.semilogy()

def CompareWithTruthSpectrum(df_residue, df_whole, v_truth_to_plot, logy=True):
    v2d_Erec = []
    v_labels = []
    E_bins = np.linspace(0,85,80)
    plt.hist(df_residue["recE"],bins=E_bins,histtype="step",
             color="black", linewidth=3, label="Residue")
    for tag_truth in v_truth_to_plot:
        v2d_Erec.append( np.array(df_whole["recE"][(df_whole["evtType"]==tag_truth) ]) )
        v_labels.append(tag_truth+"(Truth)")

    plt.hist( v2d_Erec, bins=E_bins,ls="--", stacked=True, label=v_labels)
    plt.legend()
    plt.xlabel("$E_{rec}$ [ MeV ]")
    plt.ylabel("N of Events")
    plt.title("Comparison with Truth")
    if logy:
        plt.semilogy()