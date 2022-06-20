# -*- coding:utf-8 -*-
# @Time: 2021/7/6 9:15
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: HistTools.py
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

def GetBinCenter(h_edges):
    h_edges = np.array(h_edges)
    return (h_edges[1:]+h_edges[:-1])/2

def ReBin(h_center, h_values, n_bins_to_merge:int=2):
    h_values = np.array(h_values)
    if ( len(h_center) )%n_bins_to_merge!=0:
        print("len(h_center)%n_bins_to_merge != 0!!!!!!!!!")
        exit()
    elif len(h_center) != len(h_values):
        print("len(h_center) != len(h_values)!!!!!! Check if there is a cut!!!")
        exit()
    h_center_rebin = GetBinCenter(h_center)[::n_bins_to_merge]
    for i in range(n_bins_to_merge):
        if i == 0:
            h_values_rebin = h_values[i::n_bins_to_merge]
        else:
            h_values_rebin += h_values[i::n_bins_to_merge]
    return h_center_rebin, h_values_rebin


def GetRidOfZerosBins(h_2d:np.ndarray, h_center_x:np.ndarray):
    index_to_cut = []
    for i in range(len(h_2d)):
        if np.any(h_2d[i]!=0):
            break
        index_to_cut.append(i)
    for i_reverse in range(1,len(h_2d)):
        if np.any(h_2d[-i_reverse]!=0):
            break
        index_to_cut.append(-i_reverse)

    index_to_remain = list(set(range(len(h_2d)))-set(index_to_cut))
    return h_2d[index_to_remain], h_center_x[index_to_remain]

def GetHist2DProjectionY(h_2d_input:np.ndarray, h_edges_x:np.ndarray, h_edges_y:np.ndarray, plot=False):
    h_center_x = GetBinCenter(h_edges_x)
    h_center_y = GetBinCenter(h_edges_y)
    h_2d,h_center_x = GetRidOfZerosBins(h_2d_input, h_center_x)
    print("Using GetHist2DProjection, Attention: if a column in the middle area is all zeros, it cause an error! which can be fix by adjust number of bins")
    h_mesh = np.array(np.meshgrid(h_center_y, h_center_x))
    h_projection = np.average(h_mesh[0], weights=h_2d, axis=1)
    if plot:
        plt.step(h_center_x, h_projection,where="mid", color="black" )
    return (h_center_x, h_projection)

def RedrawHistFrom_plt_hist(hist, ax=None, *args, **kargs):
    if ax==None:
        plt.step(GetBinCenter(hist[1]), hist[0],where="mid", *args, **kargs)
    else:
        ax.step(GetBinCenter(hist[1]), hist[0],where="mid", *args, **kargs)

def PlotHistNormByHits(x, bins=None, ax=None ,*args, **kargs):
    hist = np.histogram(x, bins=bins)
    hist_norm = (hist[0]/len(x), hist[1])
    RedrawHistFrom_plt_hist(hist_norm, ax,  *args, **kargs)
    return hist_norm

def PlotHistNormByMax(x, bins=None, ax=None, divide_binWidth=False,plot=True, return_errorbar=False ,*args, **kargs):
    hist = np.histogram(x, bins=bins)
    hist_error  = np.sqrt(hist[0])
    if divide_binWidth:
        hist_norm = ( (hist[0]/np.diff(hist[1])) / max(hist[0]/np.diff(hist[1])) , hist[1])
        hist_error_norm = ( (hist_error/np.diff(hist[1])) / max(hist[0]/np.diff(hist[1])) , hist[1])
    else:
        hist_norm = (hist[0]/max(hist[0]), hist[1])
        hist_error_norm = (hist_error/max(hist[0]), hist[1])
    if plot:
        RedrawHistFrom_plt_hist(hist_norm, ax,  *args, **kargs)
    if return_errorbar:
        return hist_norm, hist_error_norm[0]
    else:
        return hist_norm

def DfHistPlotNormByMax(data:pd.DataFrame, x:str,hue:str,plot_ratio=True,hue_order=None,
                        ratio_base=None, dict_cut_for_df:dict=None,key_cut:str="",*args, **kargs):
    """

    :param data:
    :param x: key in df
    :param hue:  key in df
    :param plot_ratio:
    :param hue_order:
    :param ratio_base: one key in data[hue] which is to set ratio criteria

    :param key_cut: key in data
    :param dict_cut_for_df: for example: {key1:[True, False, ...True]} or {key1:data[data[key_cut]<cut]}, key1 is in data[hue]
    note: key_cut and dict_cut_for_df should be input simultaneously

    :param args:
    :param kargs:
    :return:
    """
    import seaborn as sns
    def GetIndexCut(oneType):
        # Set Cut for Dataset
        if dict_cut_for_df is None or oneType not in dict_cut_for_df.keys():
            index_cut = (data[hue] == oneType)
        else:
            index_cut = (data[hue] == oneType) & (data[key_cut]<dict_cut_for_df[oneType][1]) &\
                        (data[key_cut] > dict_cut_for_df[oneType][0])
        return index_cut

    if (dict_cut_for_df is None and key_cut!="") or (not dict_cut_for_df is None and key_cut==""):
        print("ERROR:\tkey_cut and dict_cut_for_df should be input simultaneously!!")
        return 1

    if "ax" in kargs.keys():
        plot_ratio = False

    if hue_order is None:
        hue_order = set(np.array(data[hue]))
    if plot_ratio:
        f, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True, figsize=(8,8))
        plt.subplots_adjust(wspace=0, hspace=0)
    v_colors = sns.color_palette("bright")[:len(hue_order)]


    if not ratio_base is None:
        index_cut = GetIndexCut(ratio_base)
        df_base = data[index_cut]
        if (not dict_cut_for_df is None) and (ratio_base in dict_cut_for_df.keys()):
            df_base
        h, h_error = PlotHistNormByMax(data[index_cut][x],
                              label=ratio_base,plot=False,return_errorbar=True,
                              *args, **kargs)
        bin_center = GetBinCenter(h[1])
        hist_base = h[0]
        hist_base_error = h_error

    for i,oneType in enumerate(hue_order):
        index_cut = GetIndexCut(oneType)

        color = v_colors[i]
        if plot_ratio:
            h, h_error = PlotHistNormByMax(data[index_cut][x], label=oneType, color=color,
                                           return_errorbar=True, ax=ax0, linewidth=2,*args, **kargs)
            if ((ratio_base is None) and (i==0)) or (ratio_base==oneType):
                bin_center = GetBinCenter(h[1])
                hist_base = h[0]
                hist_base_error = h_error
                ax1.set_ylabel(r"$\frac{h-h_{"+str(oneType)+"}}{h_{"+str(oneType)+"}}$")
            else:
                ratio_error = np.sqrt( (hist_base**2*h_error**2 + h[0]**2*hist_base_error**2)/hist_base**4 )
                ax1.errorbar(bin_center, (h[0]-hist_base)/hist_base,xerr=np.diff(h[1])/2,yerr=ratio_error,color=color,
                            ecolor=color, markersize=5,ls="none",
                            capthick=1,elinewidth=1, linewidth=1)
        else:
            h = PlotHistNormByMax(data[index_cut][x], label=oneType, color=color, *args, **kargs)

    if "ax" in kargs.keys():
        kargs["ax"].legend(title=hue)
    elif plot_ratio:
        ax0.legend(title=hue)
        return (ax0, ax1)
    else:
        plt.legend(title=hue)


def GetMaxArgOfHist(v_data, bins=100):
    hist =  np.histogram(v_data, bins=bins)
    return GetBinCenter(hist[1])[np.argmax(hist[0])]

def GetAlignValue(v_time_to_align, v_time_criteria,v_charge_criteria, bins, ratio_threshold=0.2,align_method="peak"):
    h_time_to_align,_ = np.histogram(v_time_to_align, bins)
    h_time_criteria,_ = np.histogram(v_time_criteria, bins, weights=v_charge_criteria)
    bins_center = GetBinCenter(bins)
    if align_method == "peak":
        return bins_center[ np.argmax(h_time_to_align)]-bins_center[np.argmax(h_time_criteria) ]
    elif align_method == "threshold":
        return bins_center[ np.where( h_time_to_align> ratio_threshold*max(h_time_to_align) )[0][0] ] -\
            bins_center[ np.where( h_time_criteria> ratio_threshold*max(h_time_criteria) )[0][0] ]


if __name__ == '__main__':
    hist = plt.hist([1,2,35,3,5,6,1])
    plt.figure()
    RedrawHistFrom_plt_hist(hist)
    plt.show()