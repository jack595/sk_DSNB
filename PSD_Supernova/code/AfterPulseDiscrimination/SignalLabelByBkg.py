# -*- coding:utf-8 -*-
# @Time: 2022/3/22 14:44
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: SignalLabelByBkg.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from LoadMultiFiles import LoadOneFileUproot, LoadOneFile
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

    with np.load("/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/code/index_SigMixInBkg.npz", allow_pickle=True) as f:
        index_sig_labelBy_bkg = f["index"]
        index_fileNo  = f["fileNo"]

    index_sig_labelBy_bkg = index_sig_labelBy_bkg[ index_fileNo==0 ]

    from collections import Counter


    n_evts_plot = 10

    v_AP_Count = []
    v_SN_Count = []
    for i in tqdm.tqdm( index_sig_labelBy_bkg ):
        chain.GetEntry(i)
        print(chain.evtType, dir_map["evtType"][i])
        v_hitType =  np.array( chain.HitTypeTruth)
        counter = Counter( v_hitType )
        n_total_hits = len( v_hitType )
        v_AP_Count.append( counter["AfterPulse"]/n_total_hits )
        v_SN_Count.append( counter["SN"]/n_total_hits )
    print("AfterPulse:\t", v_AP_Count)
    print("SN:\t",v_SN_Count)
    np.savez("ratio_PulseType.npz", v_AP_Count=v_AP_Count, v_SN_Count=v_SN_Count)


        # break
    # np.savez("DNCount.npz", DN_count=v_DN_Count,
    #          TriggerTimeInterval=dir_features['TriggerTimeIntervalWithFilter'][index_sig_labelBy_bkg])