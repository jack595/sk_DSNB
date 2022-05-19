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
import plotly.figure_factory as ff
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def PlotTimeSeries(df_map, evtID_start, evtID_end, FV_cut, focus_evtID=None ,key_y_plot="R", key_size_plot="recE",
                   deltaT_focus_table=50, plot_table=False):
    # Get radius to do FV_cut
    if "R" not in df_map.columns:
        df_map["R"] = GetR3_XYZ( df_map["recX"],df_map["recY"], df_map["recZ"] )**(1/3)/1e3

    # Set Tags for figure
    if "TagIBD" not in df_map.columns:
        df_map["TagIBD"] = ( df_map["TagIBDp"].replace({-1:0})+df_map["TagIBDd"].replace({-1:0})*2 + np.array( (df_map["TagIBDd"]!=1) & (df_map["R"]>FV_cut),int)*9 +
                         np.array((df_map["TagAP"]==1),int)*11)\
        .replace({0:"Untagged", 1:"TagIBDp", 2:"TagIBDd", 9:"Cut By FV_Cut",11:"TagAP"})
        # df_map["TagIBD"] = df_map["TagIBD"].replace({0:"Untagged", 1:"TagIBDp", 2:"TagIBDd", 9:"Cut By FV_Cut"})

    # Set df_map to interest region
    df_map_set_index_evtID = df_map.set_index("evtID")
    df_map_tmp = df_map_set_index_evtID[(df_map_set_index_evtID["TriggerTime"] > df_map_set_index_evtID.loc[evtID_start]["TriggerTime"]) &
                        (df_map_set_index_evtID["TriggerTime"] < df_map_set_index_evtID.loc[evtID_end]["TriggerTime"]+1e1) ]
    # df_map_tmp = df_map_tmp[df_map_tmp["TagAP"]==0]

    # Plot events
    if focus_evtID!=None:
        triggerTime_focus = df_map_tmp.loc[focus_evtID]["TriggerTime"]
        df =  df_map[(df_map["TriggerTime"]>triggerTime_focus-deltaT_focus_table)&
                        (df_map["TriggerTime"]<triggerTime_focus+deltaT_focus_table)]\
                        [["evtID","evtType",  "ratioSN","ratioAP", "R", "fileNo","detIDs","detID","TriggerTime", "TagIBD","recE"]]\
                        .set_index("evtID")
        if plot_table:
            fig_table, ax =plt.subplots(figsize=(12,8))
            ax.axis('tight')
            ax.axis('off')
            df = df.round(3)
            df["recE"] = df["recE"].apply(lambda x :np.round(x, decimals=3))
            the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
            # the_table.auto_set_font_size(False)
            # the_table.set_fontsize(8)
        else:
            display(df)

        fig = plt.figure()
        plt.scatter(df_map_tmp.loc[focus_evtID]["TriggerTime"], df_map_tmp.loc[focus_evtID][key_y_plot],
                    s=100, facecolors='none',edgecolor="red")

    v_evtType = ["IBDp","pES", "IBD","IBDd", "eES", "C12", "B12", "AfterPulse","N12", "pileUp" ]
    palette =  dict( zip(v_evtType, sns.color_palette("bright")[:len(v_evtType)]) )

    sns.scatterplot(x="TriggerTime", y=key_y_plot, data=df_map_tmp, hue="evtType",
                    style="TagIBD", size=key_size_plot, sizes=(50, 200),palette=palette,
                    style_order=["Untagged", "Cut By FV_Cut", "TagIBDp", "TagIBDd","TagAP"])


    # Plot selected pair
    index_IBDd = (df_map_tmp["TagIBDd"]==1)

    v2d_time = np.array( [np.array(df_map_tmp[index_IBDd]["TriggerTime"]) ,np.array(df_map.set_index("evtID").loc[ df_map_tmp[index_IBDd]["IBDSource"] ]["TriggerTime"]) ] ).T
    v2d_recE = np.array( [np.array(df_map_tmp[index_IBDd][key_y_plot]), np.array(df_map.set_index("evtID").loc[ df_map_tmp[index_IBDd]["IBDSource"] ][key_y_plot]) ] ).T
    for time, recE in zip(v2d_time, v2d_recE):
        plt.plot( time, recE, ls="--" )
        

    # Plot MC Truth Pair
    df_map_reset_index = df_map.set_index(["fileNo","detID"]).sort_index()
    for index, row in df_map_tmp[(df_map_tmp["evtType"]=="IBDd") |(df_map_tmp["evtType"]=="IBD")].iterrows():
        fileNo = row["fileNo"]
        detID = row["detID"]
        if detID == -1:
            continue
        df_slide = df_map_reset_index.loc[ (fileNo, detID) ]
        row_IBDp = df_slide[ (df_slide["evtType"] =="IBDp") | (df_slide["evtType"] =="IBD")]
        if len(row_IBDp)>0:
            plt.plot( [row["TriggerTime"], row_IBDp["TriggerTime"]], [row[key_y_plot], row_IBDp[key_y_plot]], linewidth=1 )

    plt.title(f"EvtID: ({evtID_start},{evtID_end})")
    plt.legend(bbox_to_anchor=(1,1))
    # plt.legend(["black dotted line"], handlelength=3)
    plt.xlim(df_map_set_index_evtID.loc[evtID_start]["TriggerTime"],df_map_set_index_evtID.loc[evtID_end]["TriggerTime"])
    if plot_table:
        return fig, fig_table
    else:
        return fig

