# -*- coding:utf-8 -*-
# @Time: 2022/5/14 9:59
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FindIBDPartner.py
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from IPython.display import display

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
def UpdateDetID(df_map:pd.DataFrame):
    df_map_reset_index = df_map.set_index(["fileNo","detID"]).sort_index()
    for index, row in df_map[ (df_map["detID"]==-1) & ( (df_map["evtType"]=="IBDd") | (df_map["evtType"]=="IBDp") | (df_map["evtType"]=="IBD") )].iterrows():
        if len(row["detIDs"]) >0:
            for detID in row["detIDs"]:
                if detID not in df_map_reset_index.loc[ row["fileNo"] ].index:
                    continue
                v_evtType = list( df_map_reset_index.loc[ (row["fileNo"],detID) ]["evtType"])

                if ( ("IBDp" in v_evtType) and ("IBDd" in v_evtType) ) or ("IBD" not in "".join(v_evtType) ) :
                    continue
                df_map.at[index, "detID"] = np.array( df_map.set_index(["fileNo","detID"]).loc[ (row["fileNo"],detID ) ].index.get_level_values("detID") )[0]
    return df_map.set_index(["fileNo","detID"]).sort_index()

def FindIBDPartner(df_interest:pd.DataFrame, df_total:pd.DataFrame):
    df_total_reset_index = UpdateDetID(df_total)

    dir_evtType_pairs = {"IBDp":"IBDd", "IBDd":"IBDp"}
    v_truth_index_partner = np.ones(len(df_interest))*-1
    for i, (index, row) in enumerate(df_interest.iterrows()):
        for detID in row["detIDs"]:
            try:
                df_tmp = df_total_reset_index.loc[(row["fileNo"], detID)]
            except KeyError:
                pass
            df_truth_partner = df_tmp[df_tmp["evtType"]==dir_evtType_pairs[row["evtType"]]]
            if len(df_truth_partner)>0:
                if len(df_truth_partner["evtID"])==1:
                    v_truth_index_partner[i] = int(df_truth_partner["evtID"])
                else:
                    v_truth_index_partner[i] = -2
                    print("###############")
                    display(row)
                    display(df_truth_partner)
                    # break
            elif len(df_tmp[df_tmp["evtType"]=="IBD"]):
                v_truth_index_partner[i] = int(df_tmp[df_tmp["evtType"]=="IBD"]["evtID"])
            else:
                pass
    df_interest["IBDPartner"] = v_truth_index_partner


