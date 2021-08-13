# -*- coding:utf-8 -*-
# @Time: 2021/7/6 9:15
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: HistTools.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

def GetBinCenter(h_edges):
    return (h_edges[1:]+h_edges[:-1])/2

def GetHist2DProjectionY(h_2d:np.ndarray, h_edges_x:np.ndarray, h_edges_y:np.ndarray, plot=False):
    h_center_x = GetBinCenter(h_edges_x)
    h_center_y = GetBinCenter(h_edges_y)
    h_mesh = np.array(np.meshgrid(h_center_y, h_center_x))
    h_projection = np.average(h_mesh[0], weights=h_2d, axis=1)
    if plot:
        plt.step(h_center_x, h_projection,where="mid", color="black" )
    return (h_center_x, h_projection)