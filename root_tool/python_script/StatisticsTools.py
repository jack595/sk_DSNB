# -*- coding:utf-8 -*-
# @Time: 2022/3/22 21:52
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: StatisticsTools.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def GetEfficiencySigma(eff, n_samples):
    relative_sigma_eff = np.sqrt(eff*(1-eff)*n_samples) / n_samples
    return relative_sigma_eff