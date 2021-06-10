# -*- coding:utf-8 -*-
# @Time: 2021/5/28 14:34
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotMultiHistToH2D.py
import matplotlib.pylab as plt
import numpy as np
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
from matplotlib.colors import LogNorm

def GetBinCenter(h_edges):
    return (h_edges[1:]+h_edges[:-1])/2

def TurnMultiHistToH2D(v_hist:np.ndarray, h_edges:np.ndarray, down_ylimit=0., up_ylimit=1.0):
    bins_center = GetBinCenter(h_edges)
    h_2d = np.array([])
    xedges, yedges = np.array([]), np.array([])
    for i, hist in enumerate(v_hist):
        if i == 0:
            h_2d, xedges, yedges  = np.histogram2d(bins_center,hist,
                                                bins=(h_edges, np.linspace(down_ylimit, up_ylimit, len(h_edges))))
        else:
            h_2d_add, xedges_add, yedges_add  = np.histogram2d(bins_center,hist,
                                                bins=(h_edges, np.linspace(down_ylimit, up_ylimit, len(h_edges))))
            h_2d += h_2d_add
    return (h_2d, xedges, yedges)

def PlotHist2D(h2d:np.ndarray, xedges:np.ndarray, yedges:np.ndarray, log:bool=True, fig=None, cmap="Blues"):
    h_2d_plot = h2d.T
    X, Y = np.meshgrid(xedges, yedges)
    if fig == None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    if log:
        im = ax.pcolormesh(X, Y, h_2d_plot, cmap=cmap, norm=LogNorm())
    else:
        im = ax.pcolormesh(X, Y, h_2d_plot, cmap=cmap)
    plt.colorbar(im)