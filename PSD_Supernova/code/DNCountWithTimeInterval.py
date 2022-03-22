# -*- coding:utf-8 -*-
# @Time: 2022/3/21 11:31
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: DNCountWithTimeInterval.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
from LoadMultiFiles import LoadOneFileUproot,LoadOneFile
import tqdm

if __name__ == "__main__":
    import ROOT

    dir_map = LoadOneFileUproot(f"/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/tag_event/root/sn_tag_0.root",
                                            name_branch='evtTruth',
                                            return_list=False)

    dir_features = LoadOneFile("/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/run_ExtractFeatures_AfterPulse/features_noShift_0__full.npz",
                               key_in_file="dir_variables", whether_use_filter=False)

    chain = ROOT.TChain("evtTruth")
    chain.Add("/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/tag_event/sn_tag_0_save_TimeTruth.root")

    from collections import Counter

    index_tag = np.where( dir_map["evtType"]=="AfterPulse" )[0]
    print( dir_map["evtType"][index_tag])

    n_evts_plot = 10

    v_DN_Count = []
    for i in tqdm.tqdm( index_tag ):
        chain.GetEntry(i)
        v_DN_Count.append( Counter( np.array( chain.HitTypeTruth) )["AfterPulse"] )
    np.savez("DNCount.npz", DN_count=v_DN_Count,
             TriggerTimeInterval=dir_features['TriggerTimeIntervalWithFilter'][index_tag])

    # plt.scatter(  dir_features['TriggerTimeIntervalWithFilter'][index_tag][:n_evts_plot] , v_DN_Count)
    # plt.xlabel("Trigger Time Interval [ ns ]")
    # plt.ylabel("$N_{DarkNoise}$")
    # plt.semilogx()
    # plt.show()

