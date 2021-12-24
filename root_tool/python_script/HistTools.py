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

def RedrawHistFrom_plt_hist(hist, *args, **kargs):
    plt.step(GetBinCenter(hist[1]), hist[0],where="mid", *args, **kargs)

if __name__ == '__main__':
    hist = plt.hist([1,2,35,3,5,6,1])
    plt.figure()
    RedrawHistFrom_plt_hist(hist)
    plt.show()