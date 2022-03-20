# -*- coding:utf-8 -*-
# @Time: 2021/6/1 15:06
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: NumpyTools.py
import matplotlib.pylab as plt
import numpy as np
import random
from copy import copy
from LoadMultiFiles import MergeEventsDictionary
import math

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
def ReBin(v:np.ndarray):
    length_v = len(v)
    v_odd = v[0:length_v:2]
    v_even = v[1:length_v:2]
    print(v_odd)
    print(v_even)

def AlignRisingEdge(v_to_align:np.ndarray, threshold:float, i_loc_aligned=10):
    i_over_threshold = np.where(v_to_align>threshold)[0][0]-i_loc_aligned
    if i_over_threshold>=len(v_to_align):
        print(f"ERROR!:Cannot get the rising edge using threshold ({threshold})")
        exit()
    v_return = v_to_align
    if i_over_threshold>0:
        v_return = np.delete(v_to_align, np.arange(i_over_threshold))
        v_return = np.concatenate((v_return, np.zeros(i_over_threshold)))
    elif i_over_threshold < 0:
        i_append = -i_over_threshold
        v_return = np.concatenate((np.zeros(i_append), v_to_align))[:len(v_to_align)]
    return v_return

def Replace( array_to_replace, dir_map_to_replace:dict=None):
    if dir_map_to_replace is None:
        dir_map_to_replace = {"alpha":0, "e-":1}
    replacer = dir_map_to_replace.get
    return np.array([replacer(item, item) for item in array_to_replace])

def AlignEvents(v_dict):
    """

    :param v_dict: list for events dictionaries, whose each item should have the same length
    we will make the input dictionaries got the same length,
    :return: the dataset have been cut off, which might be useful in testing stage
    """
    one_key = list(v_dict[0].keys())[0]
    n_events_to_align = len(v_dict[0][one_key])
    for dir in v_dict[1:]:
        if n_events_to_align > len(dir[one_key]):
            n_events_to_align = len(dir[one_key])

    # Align events dict
    dir_cut_off = {}
    for dir in v_dict:
        for key in dir.keys():
            if len(dir[key]) > n_events_to_align:
                if key not in dir_cut_off:
                    dir_cut_off[key] = copy(dir[key][n_events_to_align:])
                else:
                    dir_cut_off[key] =  MergeEventsDictionary([dir_cut_off[key], copy(dir[key][n_events_to_align:])])
            dir[key] = dir[key][:n_events_to_align]
    return (n_events_to_align,dir_cut_off)

def SplitEventsDict(dir_events:dict, ratio_to_split_train:float=0.5):
    one_key = list(dir_events.keys())[0]
    n_events = len(dir_events[one_key])
    n_events_for_train = int(n_events*ratio_to_split_train)
    dir_events_train = {}
    dir_events_test = {}
    for key in dir_events.keys():
        dir_events_train[key] = dir_events[key][:n_events_for_train]
        dir_events_test[key] = dir_events[key][n_events_for_train:]
    return dir_events_train, dir_events_test

def ShuffleDataset(dir_events:dict):
    n_events = len(dir_events[list(dir_events.keys())[0]])

    # Shuffle Events
    index_shuffle = random.shuffle(np.arange(n_events))
    for key in dir_events.keys():
        dir_events[key] = dir_events[key][index_shuffle][0]

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def weighted_mean(var, wts):
    """Calculates the weighted mean"""
    return np.average(var, weights=wts)


def weighted_variance(var, wts):
    """Calculates the weighted variance"""
    return np.average((var - weighted_mean(var, wts))**2, weights=wts)


def weighted_skew(var, wts):
    """Calculates the weighted skewness"""
    return (np.average((var - weighted_mean(var, wts))**3, weights=wts) /
            weighted_variance(var, wts)**(1.5))

def weighted_kurtosis(var, wts):
    """Calculates the weighted skewness"""
    return 1./( len(var)-1 ) * (  np.dot((var - weighted_mean(var, wts))**4, wts) /
            weighted_variance(var, wts)**(2) ) - 3

def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def GetIndexOfListTags(v_evts_tags, v_tags_get_index):
    v_evts_tags = np.array(v_evts_tags)
    index = [False]*len(v_evts_tags)
    for tag in v_tags_get_index:
        index = index|(v_evts_tags==tag)
    return index

if __name__ == '__main__':
    # Test AlignRisingEdge
    h_time = np.array([0,0,0,1,3,5,6,7,15,20,16,14,12,10,9,7,5,0,0])
    print(AlignRisingEdge(h_time, threshold=5, i_loc_aligned=10))
