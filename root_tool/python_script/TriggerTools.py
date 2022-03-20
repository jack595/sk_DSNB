# -*- coding:utf-8 -*-
# @Time: 2022/3/13 20:00
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: TriggerTools.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def GetHitsCount(v_PMTID, v_Time, v_Charge, time_lowerLimit, time_upperLimit, strategy="nPMT_w/o_repeat"):
    index = (v_Time<time_upperLimit) & (v_Time>time_lowerLimit)
    if np.any( index ) == False:
       return 0
    else:
        if strategy=="nPMT_w/o_repeat":
            return len ( set( v_PMTID[ index ] ) )
        elif strategy == "nPMT_w/_repeat":
            return len (  v_PMTID[ index ]  )
        elif strategy == "Charge":
            return np.sum(v_Charge[index])
        else:
            raise ValueError("Keyword strategy not in the list of options!!!!!!!")

def GetArrayNPMTsFired(v_Time, v_PMTID, v_Charge=None, strategy="nPMT_w/o_repeat", win_width = 80,
                       step_slide = 16, time_max=1000):
    v_time_lowerLimit = np.arange(4, time_max, step_slide)
    v_time_upperLimit = v_time_lowerLimit+win_width

    v_nPMTsFired = []
    for time_lowerLimit, time_upperLimit in zip( v_time_lowerLimit, v_time_upperLimit ):
        v_nPMTsFired.append( GetHitsCount(v_PMTID, v_Time, v_Charge,
                        time_lowerLimit, time_upperLimit , strategy=strategy) )
    return v_time_lowerLimit, v_nPMTsFired
