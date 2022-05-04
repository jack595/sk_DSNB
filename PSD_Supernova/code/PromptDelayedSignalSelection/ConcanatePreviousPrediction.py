# -*- coding:utf-8 -*-
# @Time: 2022/4/21 22:33
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: ConcanatePreviousPrediction.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from LoadMultiFiles import LoadOneFileUproot
def GetVertexR(df):
    return np.sqrt(df["recX"]**2+df["recY"]**2+df["recZ"]**2)
import pandas as pd

def GetSelectedDataframe(path_evtTruth="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/tag_event/root/sn_tag_0.root",
                         path_AP="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/AfterPulsePrediction/root/TagAfterPulse_0.root",
                         path_PSD="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/PSD/root/user_PSD_0_SN.root",
                         path_IBD="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/code/PromptDelayedSignalSelection/root_PromptDelayedSelection/IBD_0_optimized.root",
                         path_CC="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/code/PromptDelayedSignalSelection/root_PromptDelayedSelection/CC_0_optimized.root",
                         path_Singles="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/code/PromptDelayedSignalSelection/root_PromptDelayedSelection/Isolation_0_optimized.root",
                         inf_from_evtType=None, load_isolationResult=False):
    dir_map = LoadOneFileUproot(path_evtTruth,
                                name_branch="evtTruth", return_list=False)
    dir_AP = LoadOneFileUproot(path_AP,
                               name_branch="AfterPulseTag", return_list=False)
    dir_PSD = LoadOneFileUproot(path_PSD,
                                name_branch="PSD", return_list=False)
    dir_IBD = LoadOneFileUproot(path_IBD,
                                name_branch="IBDSelection", return_list=False)

    dir_CC = LoadOneFileUproot(path_CC,
                                name_branch="CCSelection", return_list=False)



    df_AP = pd.DataFrame.from_dict(dir_AP)
    df_map = pd.DataFrame.from_dict(dir_map)

    # Get Vertex Radius of events
    v_R = GetVertexR(df_map)
    df_map["R"] = v_R
    if inf_from_evtType == None:
        df_map = pd.concat( (df_map, df_AP["TagAP"]),axis=1)

    else:
        df_map = pd.concat( (df_map[inf_from_evtType], df_AP["TagAP"]),axis=1)

    dir_IBD.pop("evtID")
    df_IBD = pd.DataFrame.from_dict(dir_IBD)
    df_map = pd.concat( (df_map, df_IBD), axis=1 )

    df_PSD = pd.DataFrame.from_dict(dir_PSD)
    df_PSD = df_PSD.rename({"evtType":"TagPSD"},axis=1)
    df_map = pd.concat( (df_map, df_PSD),axis=1)

    df_CC = pd.DataFrame.from_dict(dir_CC)
    df_CC.pop("evtID")
    df_map = pd.concat( (df_map, df_CC), axis=1)

    if load_isolationResult:
        dir_IsolationTag = LoadOneFileUproot( path_Singles, name_branch="SingleSelection", return_list=False )
        df_IsolationTag = pd.DataFrame.from_dict(dir_IsolationTag)
        df_map["TagSingle"] = df_IsolationTag["TagSingle"]

    return df_map