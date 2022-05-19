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
import more_itertools as mit
from copy import copy
from matplotlib.colors import LogNorm

from wavedumpReader import DataFile
def WaveDumpReader(path_file:str, nEvts:int=-1, return_Dataframe=True):
    """

    :param path_file: output from DT5751, binary file
    :param nEvts: N of events to be loaded
    :param return_Dataframe: return dataframe or dict
    :return:
    """
    reader = DataFile(path_file)
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

###################### Event Display ######################################
def PlotWaveformHist2d( v2d_waveform:np.ndarray, bins=None, show_h2d=True,
                        logz=True, ax=None, fig=None):
    v2d_waveform_aligned = []
    v2d_time = []
    if ax is None:
        fig, ax = plt.subplots(1)
    if (not ax is None) and (fig is None):
        print("PlotWaveformHist2d suppose ax and fig are input simultaneously")
        exit(0)

    for wave in v2d_waveform:
        # Align Peak of Waveform
        x_peak = np.argmax( wave )
        wave = wave[x_peak-200:]

        v2d_waveform_aligned.append( wave )
        v2d_time.append( np.arange(len(wave)) )
    if bins==None:
        bins = (np.arange(180.5,250.5), 100)

    if show_h2d:
        h = ax.hist2d( np.concatenate(v2d_time), np.concatenate(v2d_waveform_aligned),
                    bins=bins, norm=( LogNorm() if logz else None),
                    cmap="Blues")
        fig.colorbar(h[3], ax=ax)
    else:
        # Display waveforms in lines
        for wave in v2d_waveform_aligned[:100]:
            ax.plot(wave)

    ax.set_xlabel("Time [ ns ]")
    ax.set_ylabel("ADC")




###########################################################################



#####################  Waveform Reconstruction ############################

def Workflow_WaveformRec(df_data:pd.DataFrame, n_baseline=100, plot_check=False, negative=True,
                         threshold_times_std=4, width_threshold=3):
    """
    Waveform Reconstruction for data from WaveDumpReader
    :param df_data: data returned from WaveDumpReader
    :param n_baseline: n of points to get baseline
    :return:
    """
    SubtractBaseline(df_data, n_baseline=n_baseline, negative=negative)
    dir_TQ_pairs = {}
    for wave in df_data["waveform_sub_base"]:
        dir_TQ_pairs_aWaveform = WaveformRec(wave, n_baseline=n_baseline,plot_check=plot_check,
                                             threshold_times_std=threshold_times_std,
                                             width_threshold=width_threshold)

        # if dir_TQ_pairs is empty, initialize it keys with return dict by WaveformRec()
        if not dir_TQ_pairs:
            for key in dir_TQ_pairs_aWaveform.keys():
                dir_TQ_pairs[key] = []

        # Append reconstruct TQ into dir_TQ_pairs
        for key in dir_TQ_pairs_aWaveform.keys():
            dir_TQ_pairs[key].append( dir_TQ_pairs_aWaveform[key] )


    # Update TQ in df_data
    df_TQ = pd.DataFrame.from_dict( dir_TQ_pairs )
    df_data = df_data.join(df_TQ)
    index_withTQ = df_data.apply( lambda row: ( False if len(row["T"])==0 else True),axis=1 )

    df_data_return = df_data[index_withTQ].reset_index()

    # Add Charge_max and Width_max which is more convenient to index waveforms
    df_data_return["charge_max"] = df_data_return.apply( lambda row: max(row["Q"]),axis=1 )
    df_data_return["width_max"] =  df_data_return.apply( lambda row: max(row["width"]),axis=1 )
    df_data_return["amplitude_max"] = df_data_return.apply( lambda row: max(row["amplitude"]),axis=1 )
    df_data_return["valley_min"] = df_data_return.apply( lambda row: min(row["valley"]),axis=1 )


    return df_data_return

def GetTQArrays(df_data_signal:pd.DataFrame):
    dir_TQ_array = {}
    for key in ["T", "Q", "width", "amplitude", "valley"]:
        dir_TQ_array[key] = np.concatenate( np.array( df_data_signal[key] ) )

    return copy(dir_TQ_array)



def SubtractBaseline(df_data, n_baseline=100, negative=True):
    polarity = (-1 if negative else 1)
    v2d_subtract_baseline = []
    for i, wave in enumerate( df_data["waveform"] ):
        baseline = np.mean(wave[:n_baseline])
        wave = wave - baseline
        v2d_subtract_baseline.append( polarity * wave )
    df_data["waveform_sub_base"] = pd.Series( v2d_subtract_baseline )
    return v2d_subtract_baseline

def ExtendPeak(wave, num_index_peak, baseline):
    """
    Extend Peak width (Find where it starts from baseline and where drop into baseline)
    :param wave: 1d array waveform
    :param num_index_peak: use threshold to cut get number index of peak
    :param baseline: when the peak comes will be viewed as back to baseline
    :return:
    """
    num_index_peak_extended = num_index_peak
    i_start = num_index_peak[0]
    i_end = num_index_peak[-1]
    
    # Find when the peak start from baseline
    if i_start >=2:
        for i in range(1,i_start):
            num_index_peak_extended.append(i_start - i)
            if i_start-i-1>=0:
                if np.abs(wave[i_start-i])<=baseline and np.abs(wave[i_start-i-1])<=baseline:
                    break
            elif (i_start-i)>=0:
                if np.abs(wave[i_start-i])<=baseline:
                    break
            else:
                break

    # Find when the peak back to the baseline
    if len(wave)-i_end>=2:
        for i in range(1,len(wave)-i_end):
            num_index_peak_extended.append(i_end + i)
            if i_end+i+1<len(wave):
                if np.abs(wave[i_end+i])<=baseline and np.abs(wave[i_end+i+1])<=baseline:
                    break
            elif i_end+i<len(wave):
                if np.abs(wave[i_end + i]) <= baseline:
                    break
            else:
                break

    return sorted(num_index_peak_extended)

def WaveformRec(wave, n_baseline=100, threshold_times_std=4, width_threshold=3,
                plot_check=False):
    """
    Waveform reconstruction for one waveform
    :param wave:
    :param n_baseline:
    :param threshold_times_std:
    :param width_threshold:
    :param plot_check:
    :return:
    """
    dir_TQ_pairs = {"T":[], "Q":[], "width":[], "amplitude":[], "valley":[]}

    wave = np.array( wave )
    std_baseline = np.std(wave[:n_baseline])
    threshold = std_baseline*threshold_times_std

    num_of_lastPeak_tail = -1

    if plot_check:
        plt.plot(wave)

    # Find continuous over threshold
    num_index = np.where(wave>threshold)[0]
    for group in mit.consecutive_groups( num_index ):
        # One peak in waveform
        num_index_peak = list(group)

        num_index_peak_extended = ExtendPeak(wave, num_index_peak, std_baseline)

        # Exclude Sharp Peak (too narrow peak)
        if ( len(num_index_peak_extended) <= width_threshold) or \
            (num_index_peak[0]<num_of_lastPeak_tail):
            # num_of_lastPeak_tail record the index of last peak,
            # if last peak have included this peak,
            # this peak will be continued
            continue

        num_of_lastPeak_tail = num_index_peak_extended[-1]

        # Save information
        dir_TQ_pairs["Q"].append( np.sum(wave[num_index_peak_extended]) )
        dir_TQ_pairs["T"].append( num_index_peak_extended[0] )
        dir_TQ_pairs["width"].append( len(num_index_peak_extended) )
        dir_TQ_pairs["amplitude"].append(  np.max(wave[num_index_peak_extended]) )
        dir_TQ_pairs["valley"].append( np.min(wave[num_index_peak_extended]))

        if plot_check:
            plt.plot(num_index_peak_extended, wave[num_index_peak_extended])
    return copy( dir_TQ_pairs )

###################################################################################################



        






