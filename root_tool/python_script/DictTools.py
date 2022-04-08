# -*- coding:utf-8 -*-
# @Time: 2022/3/4 17:07
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: DictTools.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def FilterEventsDict(dir_evts:dict, index:np.ndarray):
    for key in dir_evts.keys():
        dir_evts[key] = np.array(dir_evts[key])
    dir_evts_filter = {}
    for key, item in dir_evts.items():
        dir_evts_filter[key] = item[index]
    return dir_evts_filter

def RenameDict(dict:dict, key_old, key_new):
    dict[key_new] = dict.pop(key_old)