# -*- coding:utf-8 -*-
# @Time: 2022/4/18 16:06
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: DisplayIBDTimeSeries.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys
import seaborn as sns
from PlotDetectorGeometry import GetR3_XYZ
from IPython.display import display
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def PlotTimeSeries(df_map, evtID_start, evtID_end, FV_cut, key_y_plot="R", key_size_plot="recE"):
    # Get radius to do FV_cut
    df_map["R"] = GetR3_XYZ( df_map["recX"],df_map["recY"], df_map["recZ"] )**(1/3)/1e3

    # Set Tags for figure
    df_map["TagIBD"] = df_map["TagIBDp"].replace({-1:0})+df_map["TagIBDd"].replace({-1:0})*2 + np.array( (df_map["TagIBDd"]!=1) & (df_map["R"]>FV_cut),int)*9
    df_map["TagIBD"] = df_map["TagIBD"].replace({0:"Untagged", 1:"TagIBDp", 2:"TagIBDd", 9:"Cut By FV_Cut"})

    # Set df_map to interest region
    df_map_tmp = df_map.set_index("evtID").loc[evtID_start:evtID_end]

    # Plot events
    sns.scatterplot(x="TriggerTime", y=key_y_plot, data=df_map_tmp, hue="evtType",
                    style="TagIBD", size=key_size_plot, sizes=(50, 200),palette="bright",
                    style_order=["Untagged", "Cut By FV_Cut", "TagIBDp", "TagIBDd"])

    # Plot selected pair
    index_IBDd = (df_map_tmp["TagIBDd"]==1)

    v2d_time = np.array( [np.array(df_map_tmp[index_IBDd]["TriggerTime"]) ,np.array(df_map.set_index("evtID").loc[ df_map_tmp[index_IBDd]["IBDSource"] ]["TriggerTime"]) ] ).T
    v2d_recE = np.array( [np.array(df_map_tmp[index_IBDd][key_y_plot]), np.array(df_map.set_index("evtID").loc[ df_map_tmp[index_IBDd]["IBDSource"] ][key_y_plot]) ] ).T
    for time, recE in zip(v2d_time, v2d_recE):
        plt.plot( time, recE, ls="--" )

    # Plot MC Truth Pair
    df_map_reset_index = df_map.set_index(["fileNo","detID"]).sort_index()
    for index, row in df_map_tmp[index_IBDd].iterrows():
        fileNo = row["fileNo"]
        detID = row["detID"]
        if detID == -1:
            continue
        df_slide = df_map_reset_index.loc[ (fileNo, detID) ]
        row_IBDp = df_slide[ df_slide["evtType"] =="IBDp" ]
        if len(row_IBDp)>0:
            plt.plot( [row["TriggerTime"], row_IBDp["TriggerTime"]], [row[key_y_plot], row_IBDp[key_y_plot]], linewidth=1 )

    plt.legend(bbox_to_anchor=(1,1))
