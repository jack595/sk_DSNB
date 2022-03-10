# -*- coding:utf-8 -*-
# @Time: 2022/3/3 22:00
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: MultiEventsAnalysis.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def GetBuffer(v_triggerTime, i_center_event:int, length_buffer:float=1e3, return_index=True):
    """

    :param v_triggerTime: array of trigger time sequences
    :param i_center_event: choose which event to be the center of buffer
    :param length_buffer:  time length of buffer
    :return: array of time buffer satisfied time interval between center_event smaller than length_buffer, first element
            returned is the center event
    """
    v_buffer_time = []
    v_buffer_index = []
    for j in range(i_center_event+1):
        if v_triggerTime[i_center_event]-v_triggerTime[i_center_event-j]>length_buffer:
            break

        v_buffer_time.append(v_triggerTime[i_center_event-j])
        v_buffer_index.append(i_center_event-j)

        if v_triggerTime[i_center_event - j] == 0:
            break
    if return_index:
        return v_buffer_index
    else:
        return v_buffer_time

def GetTimeIntervalFilterLowE(v_triggerTime, v_Erec, threshold_Erec=12, interval_default=0,
                              Erec_lastEvt_default=0):
    interval_triggerTime = 0
    Erec_lastEvt = Erec_lastEvt_default
    if len(v_triggerTime)==1:
        interval_triggerTime = interval_default
    else:
        for i, (triggerTime, Erec) in enumerate(zip(v_triggerTime, v_Erec)):
            if i==0:
                t0 = triggerTime
                continue
            elif Erec<threshold_Erec or t0-triggerTime<0:
                continue
            interval_triggerTime = t0-triggerTime
            Erec_lastEvt = Erec
            break
    return interval_triggerTime, Erec_lastEvt