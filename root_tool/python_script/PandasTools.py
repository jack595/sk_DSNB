# -*- coding:utf-8 -*-
# @Time: 2022/2/22 18:59
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PandasTools.py
import numpy as np
import sys
from copy import copy
import matplotlib.pylab as plt

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def AlignDirElements(dir_to_align:dict):
    """
    This function is to align the items in dictionary, so that it can be converted to pandas dataframe
    :param dir_to_align: dir_example = {"A":[1,4,3], "B":"property"}
    :return: dir_example = {"A":[1,4,3], "B":["property"]*3}
    """
    dir_return = copy(dir_to_align)
    v_n_length = []
    keys_to_align = []
    for key in dir_to_align.keys():
        if isinstance(dir_to_align[key], list) or isinstance(dir_to_align[key], np.ndarray):
            v_n_length.append(len(dir_to_align[key]))
        else:
            keys_to_align.append(key)

    if len(set(v_n_length))!=1:
        print("ERROR in AlignDirElements: length of elements is not the same" )
        exit(1)

    n_evts = v_n_length[0]
    for key in keys_to_align:
        dir_return[key] = [dir_to_align[key]]*n_evts
    return dir_return

def PlotDataframeIntoPie(df, explode=None):
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    theme = plt.get_cmap('hsv')
    colors = [theme(1. * i / len(df))
                             for i in range(len(df))]
    if explode == None:
        explode = [0.01]*len(df)
    df.plot.pie(autopct='%.2f%%',colors=colors,explode=explode)

if __name__ == '__main__':
    dir = {"A":[1,3,5,5], "B":"quartz", "C":np.array([5,6, 2,5])}
    print(AlignDirElements(dir))
