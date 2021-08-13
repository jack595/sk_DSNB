# -*- coding:utf-8 -*-
# @Time: 2021/7/13 22:11
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: ReweightSamples.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

import random
def GetReweightSamples(samples_need_reweight:np.ndarray, h_distribution_before_reweight,
                       edges_distribution_before_reweight,
                       h_distribution_after_reweight):
    index_return = []
    edges_distribution_after_reweight = edges_distribution_before_reweight
    bin_width_after_reweight = np.diff(edges_distribution_after_reweight)
    center_bin = (edges_distribution_after_reweight[:-1]+edges_distribution_after_reweight[1:])/2


    # normalize distribution
    h_distribution_before_reweight = h_distribution_before_reweight/np.sum(h_distribution_before_reweight)
    h_distribution_after_reweight = h_distribution_after_reweight/np.sum(h_distribution_after_reweight)


    v_reweight_ratio = [h_distribution_after_reweight[i]/h_distribution_before_reweight[i] if h_distribution_before_reweight[i]!=0 else 0 for i in range(len(h_distribution_before_reweight))]
    scale_factor_evts = np.max(v_reweight_ratio) #because sometimes the pdf of CC is greater than pdf. of CC_samples
    #which will cause the number of selected events is out of range of existing samples.

    for i in range(len(bin_width_after_reweight)):
        index_X_bin_cut = (samples_need_reweight<center_bin[i]+bin_width_after_reweight[i]*0.5) &\
                          (samples_need_reweight>=center_bin[i]-bin_width_after_reweight[i]*0.5)
        index_number_X_bin_cut = np.where(index_X_bin_cut==True)[0]

        n_evts_need_selected = round(len(index_number_X_bin_cut)*v_reweight_ratio[i]/scale_factor_evts)
        index_number_select = random.sample( list(index_number_X_bin_cut), n_evts_need_selected)
        index_return += list(index_number_select)
    return index_return