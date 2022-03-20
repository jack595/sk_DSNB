# -*- coding:utf-8 -*-
# @Time: 2022/3/11 11:23
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: AnalysisTools.py
import matplotlib.pylab as plt
import numpy as np
from PlotTools import LegendNotRepeated
import pandas as pd
from HistTools import GetBinCenter

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys
from PlotTools import GetListOfHist2DColor
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def PlotTimeProfileAfterCut(df_variables:pd.DataFrame, bins ,index_cut,plot_hist2d=False, key_time="h_time_with_charge", v_tags=None,
                            v_colors=None, n_to_plot=20, name_pdf=""):
    if name_pdf == "":
        print_to_pdf = False
    else:
        print("Saving PDF ..........")
        print_to_pdf = True

    if "{}" not in name_pdf:
        name_pdf = name_pdf.replace('.pdf', '_{}.pdf')
    print(name_pdf)
    if v_tags is None:
        v_tags = ["AfterPulse", "eES","pES"]

    if print_to_pdf:
        dir_pdf = {tag:PdfPages(name_pdf.format(tag)) for tag in v_tags}

    if v_colors is None:
        from PlotTools import GetListOfLineColor
        v_colors = GetListOfLineColor()

    if plot_hist2d:
        dir_bins = {key:[] for key in v_tags}
        dir_hist = {key:[] for key in v_tags}
    
    bins_center = GetBinCenter(bins)

    for i_tag, tag in enumerate( v_tags ):
        if not plot_hist2d and not print_to_pdf:
            plt.figure()
        df_variables_cut = df_variables[ (index_cut) & (df_variables["tag"]==tag) ]
        for i, (index, row) in enumerate( df_variables_cut.iterrows() ):
            if print_to_pdf and not plot_hist2d:
                plt.figure()

            h_time = row[key_time]

            if plot_hist2d:
                dir_bins[tag] += list(bins_center)
                dir_hist[tag] += list(h_time/max(h_time))
            else:
                plt.plot( bins_center, h_time/max(h_time), label=tag, color=v_colors[i_tag], linewidth=0.5)


            if not plot_hist2d:
                LegendNotRepeated()
                plt.xlabel("Time [ ns ]")
                if print_to_pdf:
                    dir_pdf[tag].savefig()
                    plt.close()

            if i >n_to_plot:
                break

    if print_to_pdf:
        for tag in dir_pdf.keys():
            dir_pdf[tag].close()

    if plot_hist2d:
        v_colors_map = GetListOfHist2DColor()
        for i, tag in enumerate(dir_hist.keys()):

            if len(dir_hist[tag])==0:
                print("Length of hist in tag "+tag+" is zeros, Continue!!!!")
                continue

            plt.figure()
            plt.hist2d(dir_bins[tag], dir_hist[tag], bins=(bins, np.linspace(0,1, 100)), cmap=v_colors_map[i], norm=LogNorm())
            plt.title(tag)
            plt.colorbar()
            plt.xlabel("Time [ ns ]")

