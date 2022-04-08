# -*- coding:utf-8 -*-
# @Time: 2022/3/20 19:14
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: ExtractFeaturesToStudyTruth.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys
import pandas as pd

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
from NumpyTools import weighted_kurtosis, weighted_skew
from ExtractFeatureForAfterPulse import GetTimeInterval



def ExtractFeatureForTruth(dir_map:dict, dir_evts:dict,bins,v_tags_save=None, Ecut=None, t_length_buffer=1e6,
                           list_truth_to_save=None):
    from NumpyTools import weighted_avg_and_std
    from scipy.interpolate import interp1d
    from HistTools import GetBinCenter
    if list_truth_to_save == None:
        list_truth_to_save = [ "HitTypeTruth", "PulseTimeTruth", "TriggerTime"]
    dir_variables = {"tag":[], "R3":[],  "TotalCharge_lastEvt":[], "TriggerTimeIntervalWithFilter":[], "TotalCharge":[]}

    for name_truth in list_truth_to_save:
        dir_variables[name_truth] = []


    if v_tags_save == None:
        v_tags_save = set(dir_map["evtType"])

    dir_map["TotalCharge"] = []
    for v_time in dir_evts["h_time_with_charge"]:
        dir_map["TotalCharge"].append( np.sum( v_time) )
    dir_map["TotalCharge"] = np.array( dir_map["TotalCharge"] )

    # for time_type in ["h_time_with_charge", "h_time_without_charge"]:
    for i, tag in enumerate(v_tags_save):

        if Ecut == None:
            index_tag = (dir_map["evtType"]==tag)
        else:
            index_tag = (dir_map["evtType"]==tag) & (dir_map["recE"]<Ecut)

        for j, v_time in enumerate(dir_evts["h_time_with_charge"][index_tag]):
            dir_variables["tag"].append(tag)

            nums_index_tag = np.where(index_tag)[0]

            interval_trigger, Erec_lastEvt = GetTimeInterval(nums_index_tag[j], dir_map, t_length_buffer,
                                                             threshold_Erec=dir_map["TotalCharge"][index_tag][j]*10,
                                                             tag_energy="TotalCharge" )
            dir_variables["TriggerTimeIntervalWithFilter"].append(interval_trigger)
            # dir_variables["recE_lastEvt"].append(Erec_lastEvt)
            dir_variables["TotalCharge_lastEvt"].append(Erec_lastEvt)

            dir_variables["R3"].append( np.sqrt(dir_map["recX"][index_tag][j]**2+dir_map["recY"][index_tag][j]**2+dir_map["recZ"][index_tag][j]**2)**3/1e9)

            dir_variables["TotalCharge"].append( dir_map["TotalCharge"][index_tag][j] )


            for name_truth in list_truth_to_save:
                dir_variables[name_truth].append( np.array( dir_map[name_truth][index_tag][j]) )


        return dir_variables



