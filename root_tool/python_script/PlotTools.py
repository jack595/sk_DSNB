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
    return ["b","g","r","c","m","y","k"]
def GetListOfHist2DColor():
    return [ 'Oranges', 'Purples', 'Blues', 'Greens', 'Reds', 'Greys', ]*5
def GetRespondingColor():
    return ['grey', 'orange','purple', 'blue','green', 'red']*5

def PlotContributionOfEachArray(v2d_input:np.ndarray,label_columns="", label_index="", columns_legend:int=1,
                                show_table=False,colormap="tab20c", reverse=False):
    """

    :param v2d: v2d[0] is an array ( can be understood as an event ),
                v2d[0][:] is for contribution of each step

            colormap: options can be found in http://ipacc.ihep.ac.cn/

            label_index: for the label of first dimension, v2d[0], example :event
            label_columns: for the label of second dimension, v2d[1], example:step

    :return:
    """
    # Align the shape of arrays in second dimension
    v2d = copy(v2d_input)
    if reverse:
        for i in range(len(v2d_input)):
            v2d[i] = v2d_input[i][::-1]

        if isinstance(label_columns, list):
            label_columns = label_columns[::-1]
    v_length = [len(v_Ek) for v_Ek in v2d]
    v2d_Ek_align = np.array([np.pad(v, (0, np.max(v_length) - len(v)), 'constant') for v in v2d])
    if isinstance(label_index, str) and isinstance(label_columns, str):
        df_Ek = pd.DataFrame(v2d_Ek_align, columns=[f"{label_columns} {i}" for i in range(np.max(v_length))],
                         index=[f"{label_index} {i}" for i in range(len(v2d_Ek_align))])
    elif isinstance(label_index, list) and isinstance(label_columns, list):
        df_Ek = pd.DataFrame(v2d_Ek_align, columns=label_columns,
                             index=label_index)
    else:
        print("Both label_columns and label_index should be str or list!!!!!!!!!")
        exit(1)

    if show_table:
        display(df_Ek)
    df_Ek.plot.bar(stacked=True,colormap=colormap)
    plt.legend(bbox_to_anchor=(1.0, 1.0),ncol=columns_legend)


def LegendNotRepeated(*args, **kwargs):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), *args, **kwargs)

def AutoGetColorWithALotOfLine(i_iterate, i_start=5):
    """
    This function is used to get color when using science style and number of lines is more than 5,
    it will return extra colors so that make the colors not repeat
    :param i_iterate: the i-th line
    :return: color for plt.plot
    """
    v_colors = [ "purple","blue", "black", "c"]
    if i_iterate>i_start and i_iterate <= i_start+len(v_colors) :
        color = v_colors[i_iterate-i_start-1]
    else:
        color = None
    return color