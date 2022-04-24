# -*- coding:utf-8 -*-
# @Time: 2022/4/24 15:45
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: DataReader.py
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from wavedumpReader import DataFile
def WaveDumpReader(path_file:str, nEvts:int=-1, with_header=True,
                   return_Dataframe=True):
    reader = DataFile(path_file)
    v2d_waveforms = []
    dir_data = {"boardID":[],  "filePath":[],"channel":[], "pattern":[], "eventCounter":[],
                "triggerTimeTag":[], "triggerTime":[], "waveform":[]}
    i_event = 0
    while True:
        try:
        # if True:
            trigger = reader.getNextTrigger()
            dir_data["waveform"].append( trigger.trace)
            dir_data["boardID"].append( reader.boardId)
            dir_data["filePath"].append( trigger.filePos)
            dir_data["channel"].append( trigger.channel)
            dir_data["pattern"].append( trigger.pattern)
            dir_data["eventCounter"].append( trigger.eventCounter)
            dir_data["triggerTimeTag"].append( trigger.triggerTimeTag)
            dir_data["triggerTime"].append( trigger.triggerTime)
        except AttributeError:
            break

        # if nEvts==-1, loops over the whole samples
        if (nEvts!=-1) & (i_event>nEvts):
            break
        i_event += 1

    if return_Dataframe:
        return pd.DataFrame.from_dict(dir_data)
    else:
        return dir_data

    # plt.plot(trigger.trace)


    # with open(path_file, "rb") as f:
    #     if with_header:
    #         i0, i1, i2, i3, i4, i5 = np.fromfile(f, dtype='I', count=6)
    #     else:
    #         print("WaveDump Reader without header haven't been implemented yet.")
    #         exit(0)
    #
    #     eventSize = (i0 - 24) // 2
    #     boardId = i1
    #     pattern = i2
    #     channel = i3
    #     eventCounter = i4
    #     triggerTimeTag = i5
#