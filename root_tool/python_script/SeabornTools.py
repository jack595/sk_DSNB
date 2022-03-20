# -*- coding:utf-8 -*-
# @Time: 2022/3/14 15:12
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: SeabornTools.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

import seaborn as sns

def JointPlot(data, x:str, y:str, hue:str, xlabel="", ylabel=""):
    g = sns.JointGrid(x=x,y=y,data=data,hue=hue)
    if xlabel == "":
        xlabel = x
    if ylabel == "":
        ylabel = y
    g.set_axis_labels(xlabel, ylabel)
    g.plot_marginals(sns.histplot, element="step", fill=False)
    g.plot_joint(sns.scatterplot,s=3 )
    return g