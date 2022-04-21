# -*- coding:utf-8 -*-
# @Time: 2021/11/8 20:04
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotDetectorGeometry.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from mpl_toolkits.mplot3d.axes3d import Axes3D

def GetR3(XYZ):
    XYZ = np.array(XYZ)
    return np.sum((XYZ/1e3)**2)**(3/2)

def GetR3_XYZ(X, Y, Z):
    return ( X**2+Y**2+Z**2 )**1.5

# 3D detector sphere
def PlotBaseSphere(ax:Axes3D=None, R=17.5):
    # draw sphere
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)*R
    y = np.sin(u)*np.sin(v)*R
    z = np.cos(v)*R
    ax.plot_wireframe(x, y, z, color="black", linewidth=0.8, ls="--")

def PlotBaseCircle(ax:plt.axis=None, R=17.5):
    circle = plt.Circle((0,0), R,color="black", fill=False,label="LS")
    if ax is None:
        fig, ax = plt.subplots()
    ax.add_patch(circle)
    R_ax = R+2.5
    ax.set_xlim(-R_ax,R_ax)
    ax.set_ylim(-R_ax,R_ax)
    ax.set_aspect(1.0)