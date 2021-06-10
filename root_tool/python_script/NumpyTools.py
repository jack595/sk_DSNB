# -*- coding:utf-8 -*-
# @Time: 2021/6/1 15:06
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: NumpyTools.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
def ReBin(v:np.ndarray):
    length_v = len(v)
    v_odd = v[0:length_v:2]
    v_even = v[1:length_v:2]
    print(v_odd)
    print(v_even)