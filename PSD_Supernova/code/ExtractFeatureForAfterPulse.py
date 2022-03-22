# -*- coding:utf-8 -*-
# @Time: 2022/3/5 9:43
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: ExtractFeatureForAfterPulse.py
import numpy as np

import sys
from scipy.stats import skew
import tqdm

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from NumpyTools import weighted_kurtosis, weighted_skew

def GetTimeInterval(i, dir_map, t_length_buffer=1e6, threshold_Erec=5, tag_energy="recE"):

    from MultiEventsAnalysis import GetBuffer,GetTimeIntervalFilterLowE
    v_buffer_time = []
    v_buffer_Erec = []
    if i == 0:
        interval_trigger = 0
        Erec_lastEvt_default = 0
    else:
        v_triggerTime = dir_map["TriggerTime"]
        index_buffer = GetBuffer(v_triggerTime,i, t_length_buffer)
        v_buffer_time = v_triggerTime[index_buffer]
        v_buffer_Erec = dir_map[tag_energy][index_buffer]
        Erec_lastEvt_default = dir_map[tag_energy][i-1]

    interval_trigger,Erec_lastEvt =GetTimeIntervalFilterLowE(v_buffer_time, v_buffer_Erec,
                                                interval_default=dir_map["TriggerTimeInterval"][i],
                                                Erec_lastEvt_default=Erec_lastEvt_default, threshold_Erec=threshold_Erec)
    if interval_trigger < 0:
        print(np.diff(v_buffer_time), v_buffer_Erec, interval_trigger)
    return interval_trigger, Erec_lastEvt


def ExtractFeature(dir_map:dict, dir_evts:dict,bins,v_tags_save=None, Ecut=None, t_length_buffer=1e6, save_h_time=False, save_truth=False ):
    from NumpyTools import weighted_avg_and_std
    from scipy.interpolate import interp1d
    from HistTools import GetBinCenter
    dir_variables = {"tag":[], "std":[], "mean":[], "max_total":[],"cumulate2":[],"cumulate5":[],"cumulate8":[],"max":[], "fluctuation":[],
                     "TriggerTimeInterval":[], "R3":[], "Kurtosis":[], "TotalCharge_lastEvt":[],"recE_lastEvt_without_filter":[],
                     "TotalCharge_lastEvt_without_filter":[], "TriggerTimeIntervalWithFilter":[],
                     "recE":[], "TotalCharge":[], "TimeOverThreshold":[],"Skewness":[]}

    if save_h_time:
        dir_variables["h_time_with_charge"] = []

    return_tags = False
    if v_tags_save == None:
        return_tags = True
        v_tags_save = set(dir_map["evtType"])

    dir_map["TotalCharge"] = []
    for v_time in dir_evts["h_time_with_charge"]:
        dir_map["TotalCharge"].append( np.sum( v_time) )
    dir_map["TotalCharge"] = np.array( dir_map["TotalCharge"] )

    bins_center = GetBinCenter(bins)


    if save_truth:
        for key in dir_map.keys():
            if "Truth" in key:
                dir_variables[key] = []


    def AppendFeatures():
       for j, v_time in tqdm.tqdm( enumerate(dir_evts[time_type][index_tag]) ):
            h_time = v_time/np.diff(bins)
            mean, std = weighted_avg_and_std( bins_center, weights=h_time/np.max(h_time) )
            dir_variables["tag"].append(dir_map["evtType"][index_tag][j])
            dir_variables["std"].append(std)
            dir_variables["mean"].append(mean)
            dir_variables["max_total"].append(np.max(h_time)/np.sum(h_time))
            dir_variables["max"].append(np.max(h_time))

            # Using cumulative curve to get PSD variable
            f_cumulate = interp1d(np.cumsum(np.diff(bins)*h_time/np.sum(np.diff(bins)*h_time)),GetBinCenter(bins))
            dir_variables["cumulate2"].append(float(f_cumulate(0.2)))
            dir_variables["cumulate5"].append(float(f_cumulate(0.5)))
            dir_variables["cumulate8"].append(float(f_cumulate(0.8)))

            # Fluctuation
            h_time_diff = np.diff(h_time/np.max(h_time))
            dir_variables["fluctuation"].append(np.sum(np.abs(h_time_diff)))

            dir_variables["TriggerTimeInterval"].append(dir_map["TriggerTimeInterval"][index_tag][j])

            nums_index_tag = np.where(index_tag)[0]

            dir_variables["recE_lastEvt_without_filter"].append(dir_map["recE"][nums_index_tag[j]-1] if dir_map["TriggerTimeInterval"][index_tag][j]!=0 else -1 )
            dir_variables["TotalCharge_lastEvt_without_filter"].append(dir_map["TotalCharge"][nums_index_tag[j]-1] if dir_map["TriggerTimeInterval"][index_tag][j]!=0 else -1 )

            interval_trigger, Erec_lastEvt = GetTimeInterval(nums_index_tag[j], dir_map, t_length_buffer,
                                                             threshold_Erec=dir_map["TotalCharge"][index_tag][j]*10,
                                                             tag_energy="TotalCharge" )
            dir_variables["TriggerTimeIntervalWithFilter"].append(interval_trigger)
            # dir_variables["recE_lastEvt"].append(Erec_lastEvt)
            dir_variables["TotalCharge_lastEvt"].append(Erec_lastEvt)

            dir_variables["R3"].append( np.sqrt(dir_map["recX"][index_tag][j]**2+dir_map["recY"][index_tag][j]**2+dir_map["recZ"][index_tag][j]**2)**3/1e9)

            index_time_cut = ( bins_center<480 )
            h_time_cut = h_time[index_time_cut]
            # h_time_cut = h_time
            # dir_variables["Kurtosis"].append( 1/(len(h_time_cut)-1) * np.sum( (h_time_cut-np.mean(h_time_cut))**4 )/np.std(h_time_cut)**4 -3 )
            dir_variables["Kurtosis"].append( weighted_kurtosis(bins_center[index_time_cut], h_time_cut))
            dir_variables["Skewness"].append( weighted_skew(bins_center[index_time_cut], h_time_cut) )

            dir_variables["recE"].append(dir_map["recE"][index_tag][j])

            dir_variables["TotalCharge"].append( dir_map["TotalCharge"][index_tag][j] )

            dir_variables["TimeOverThreshold"].append( bins_center[ h_time/max(h_time) > 0.5 ][0] )

            # Save Time Profile
            if save_h_time:
                dir_variables["h_time_with_charge"].append( h_time )

            if save_truth:
                for key in dir_map.keys():
                    if "Truth" in key:
                        dir_variables[key].append( np.array( dir_map[key][index_tag][j]) )


    # for time_type in ["h_time_with_charge", "h_time_without_charge"]:
    time_type = "h_time_with_charge"

    if not return_tags:
        for i, tag in enumerate(v_tags_save):
            if Ecut == None:
                index_tag = (dir_map["evtType"]==tag)
            else:
                index_tag = (dir_map["evtType"]==tag) & (dir_map["recE"]<Ecut)
            AppendFeatures()

    else:
        index_tag = [True]*len(dir_map["evtType"])
        AppendFeatures()



        for key in dir_variables.keys():
            if "TriggerTime" in key:
                dir_variables[key] = np.array(dir_variables[key])/1000 # us
                # dir_variables[key] /= 1000 # us
        # for key, v in dir_variables.items():
        #     print(key, len(v))

        if return_tags:
            return dir_variables, v_tags_save
        else:
            return dir_variables