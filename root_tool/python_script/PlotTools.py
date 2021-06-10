# -*- coding:utf-8 -*-
# @Time: 2021/6/8 10:08
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotTools.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
def GetListOfCmap():
    return [plt.cm.spring ,plt.cm.hot, plt.cm.winter, plt.cm.autumn, plt.cm.pink, plt.cm.cool]
