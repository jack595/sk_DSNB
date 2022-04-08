# -*- coding:utf-8 -*-
# @Time: 2022/4/1 14:30
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: TagAfterPulse.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

import argparse
from scipy.interpolate import interp1d
import pandas as pd
from LoadMultiFiles import LoadOneFileUproot
import tqdm

def GetTimeInterval(i, dir_map, t_length_buffer=1e9, threshold_Erec=5, tag_energy="recE"):

    from MultiEventsAnalysis import GetBuffer,GetTimeIntervalFilterLowE
    v_buffer_time = []
    v_buffer_Erec = []
    if i == 0:
        interval_trigger = 0
        Erec_lastEvt_default = 0
        return t_length_buffer, 0
    else:
        v_triggerTime = dir_map["TriggerTime"]
        index_buffer = GetBuffer(v_triggerTime,i, t_length_buffer)
        v_buffer_time = v_triggerTime[index_buffer]
        v_buffer_Erec = dir_map[tag_energy][index_buffer]
        Erec_lastEvt_default = dir_map[tag_energy][i-1]

    # interval_trigger,Erec_lastEvt =GetTimeIntervalFilterLowE(v_buffer_time, v_buffer_Erec,
    #                                             interval_default=dir_map["TriggerTimeInterval"][i],
    #                                             Erec_lastEvt_default=Erec_lastEvt_default, threshold_Erec=threshold_Erec)
    interval_trigger,Erec_lastEvt =GetTimeIntervalFilterLowE(v_buffer_time, v_buffer_Erec,
                                                             interval_default=t_length_buffer,
                                                             Erec_lastEvt_default=0, threshold_Erec=threshold_Erec)
    if interval_trigger < 0:
        print(np.diff(v_buffer_time), v_buffer_Erec, interval_trigger)

    if interval_trigger<0.5:
        print("Interval==0:\t",interval_trigger, Erec_lastEvt, dir_map["evtType"][index_buffer],
              dir_map["TriggerTime"][index_buffer])
    return interval_trigger, Erec_lastEvt


def cut_func(TriggerTimeInterval, TotalCharge, v_x_cut, v_y_cut):
    f = interp1d(v_y_cut, v_x_cut, fill_value="extrapolate")

    TriggerTimeInterval_cut = f(TotalCharge)

    if TriggerTimeInterval>=TriggerTimeInterval_cut:
        return 0
    else:
        return 1

def TagAfterPulse(dir_variables:dict, path_cut_csv:str, Q_cut):
    df_cut_line = pd.read_csv(path_cut_csv)
    df_variables = pd.DataFrame.from_dict(dir_variables)
    df_variables["TagAP"] = df_variables.apply( lambda row: cut_func(row["TriggerTimeIntervalWithFilter"],
                                                                   row["TotalCharge_lastEvt"], df_cut_line["x_cut"], df_cut_line["y_cut"]), axis=1)
    df_variables["TagAP(add_Q_cut)"] = df_variables.apply( lambda row: ( 0 if ( (row["TotalCharge"]>Q_cut) & (row["TagAP"]==1) ) else row["TagAP"]) , axis=1  )
    return np.array( df_variables["TagAP(add_Q_cut)"] )


def ExtractPIDVal(dir_map:dict, dir_evts:dict,t_length_buffer=1e6):
    dir_variables = {"tag":[],   "TotalCharge_lastEvt":[], "TriggerTimeIntervalWithFilter":[], "TotalCharge":[]}

    dir_map["TotalCharge"] = []
    for v_time in dir_evts["h_time_with_charge"]:
        dir_map["TotalCharge"].append( np.sum( v_time) )
    dir_map["TotalCharge"] = np.array( dir_map["TotalCharge"] )

    for i, tag in tqdm.tqdm( enumerate( dir_map["evtType"]) ):
        dir_variables["tag"].append(tag)

        interval_trigger, Erec_lastEvt = GetTimeInterval(i, dir_map, t_length_buffer,
                                                         threshold_Erec=dir_map["TotalCharge"][i]*10,
                                                         tag_energy="TotalCharge" )

        dir_variables["TriggerTimeIntervalWithFilter"].append(interval_trigger/1e3)

        dir_variables["TotalCharge_lastEvt"].append(Erec_lastEvt)


        dir_variables["TotalCharge"].append( dir_map["TotalCharge"][i] )

    return dir_variables


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Tagging AfterPulse')
    parser.add_argument("--input-PSDTools", type=str,
                        default="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/PSD/root/user_PSD_0_SN.root",
                        help="path template of input about PSDTools")
    parser.add_argument("--input-evtTruth", type=str,
                        default="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/tag_event/root/sn_tag_0.root",
                        help="path template of input about evtTruth")
    parser.add_argument("--output", "-o", type=str, default="try_0.root", help="name of outfile")
    arg = parser.parse_args()

    dir_evts = LoadOneFileUproot(arg.input_PSDTools,name_branch="evt",
                                               return_list=False)
    dir_map = LoadOneFileUproot(arg.input_evtTruth, name_branch='evtTruth',
                                              return_list=False)

    dir_variables = ExtractPIDVal(dir_map, dir_evts)
    dir_variables["TagAP"] = TagAfterPulse(dir_variables,  path_cut_csv="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/run_ExtractFeatures_AfterPulse/Cut_Parameters.csv",
                Q_cut=4000)

    # Save results into root file
    
    for key in dir_variables.keys():
        dir_variables[key] = np.array( dir_variables[key] )
    import ROOT
    from collections import Counter
    print(  Counter( list( zip(dir_variables["tag"], dir_variables["TagAP"]) ) ) )
    dir_variables.pop("tag")
    rdf = ROOT.RDF.MakeNumpyDataFrame(dir_variables)
    rdf.Snapshot("AfterPulseTag", arg.output)
    rdf.Display().Print()




