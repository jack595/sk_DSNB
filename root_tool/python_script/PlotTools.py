# -*- coding:utf-8 -*-
# @Time: 2021/6/8 10:08
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotTools.py
import matplotlib.pylab as plt
import numpy as np
from IPython.display import display
import pandas as pd
from copy import copy

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
def GetListOfCmap():
    return [plt.cm.spring ,plt.cm.hot, plt.cm.winter, plt.cm.autumn, plt.cm.pink, plt.cm.cool]
def GetListOfLineColor():
    return ["b","g","r","c","m","y","k","w"]
def GetListOfHist2DColor():
    return ['Greys', 'Oranges', 'Purples', 'Blues', 'Greens', 'Reds' ]*5
def GetRespondingColor():
    return ['grey', 'orange','purple', 'blue','green', 'red']*5

def PlotContributionOfEachArray(v2d_input:np.ndarray,label_columns="", label_index="", columns_legend:int=1,
                                show_table=False,colormap="tab20c", reverse=False):
    """

    :param v2d: v2d[0] is an array ( can be understood as an event ),
                v2d[0][:] is for contribution of each step

            colormap: options can be found in http://ipacc.ihep.ac.cn/

            label_index: for the label of first dimension, v2d[0], example :event
            label_colums: for the label of second dimension, v2d[1], example:step

    :return:
    """
    # Align the shape of arrays in second dimension
    v2d = copy(v2d_input)
    if reverse:
        for i in range(len(v2d_input)):
            v2d[i] = v2d_input[i][::-1]
    v_length = [len(v_Ek) for v_Ek in v2d]
    v2d_Ek_align = np.array([np.pad(v, (0, np.max(v_length) - len(v)), 'constant') for v in v2d])

    df_Ek = pd.DataFrame(v2d_Ek_align, columns=[f"{label_columns} {i}" for i in range(np.max(v_length))],
                         index=[f"{label_index} {i}" for i in range(len(v2d_Ek_align))])
    if show_table:
        display(df_Ek)
    df_Ek.plot.bar(stacked=True,colormap=colormap)
    plt.legend(bbox_to_anchor=(1.0, 1.0),ncol=columns_legend)


def LegendNotRepeated(*args, **kwargs):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), *args, **kwargs)
