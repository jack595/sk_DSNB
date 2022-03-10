# -*- coding:utf-8 -*-
# @Time: 2021/7/13 16:11
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: LoadMultiFiles.py
import numpy as np
import glob
from copy import copy
import uproot as up
from collections import Counter
import tqdm
import subprocess
import os
def LoadMultiFiles(name_files:str="predict_*.npz", key_in_file:str="dir_events",n_files_to_load=-1,
                   whether_use_filter=False):
    files_list = glob.glob(name_files)
    f = np.load(files_list[0],
                allow_pickle=True)
    print("key in files:\t",f.files)
    evts_0 = f[key_in_file].item()
    print("Loaded Data Keys:\t", evts_0.keys())
    evts = {}
    for key in evts_0.keys():
        evts[key] = []
    if n_files_to_load!=-1:
        files_list = files_list[:n_files_to_load]
    for file in files_list:
        load_into_evts = True
        with np.load(file, allow_pickle=True) as f:
            evts_load = f[key_in_file].item()

            ############# Filter abnormal file##########
            # For saving time we can turn off this section
            if whether_use_filter:
                if not evts_load:
                    load_into_evts = False
                else:
                    length_evts_0 = len(list(evts_load.items())[0][1])
                    for key in evts_load.keys():
                        if not list(evts_load[key]) :
                            load_into_evts = False
                        else:
                            # print(length_evts_0, len(evts_load[key]))
                            if length_evts_0 != len(evts_load[key]):
                                load_into_evts = False
            ##############################################

            if load_into_evts:
                for key in evts_load.keys():
                    evts[key].extend(evts_load[key])
            else:
                print(f"{file} will be continue")
    for key in evts.keys():
        try:
            evts[key] = np.array(evts[key])
        except Exception:
            print("Something go wrong, continue!")
            continue
    return evts
def MergeEventsDictionary(v_dir:list):
    keys_one_dir = v_dir[0].keys()
    for dir in v_dir:
        if keys_one_dir != dir.keys():
            print(keys_one_dir, " !=  ", list(dir.keys()))
            print("ERROR: Input dictionaries should have the same keys!!!!!!!!!!!!")
            exit(1)

    # Initialization
    dir_return = copy(v_dir[0])

    # Merge into one dictionary
    for key in dir_return.keys():
        dir_return[key] = list(dir_return[key])
    for dir in v_dir[1:]:
        for key in dir.keys():
            dir_return[key].extend(list(dir[key]))
    for key in dir_return.keys():
        dir_return[key] = np.array(dir_return[key])
    return dir_return

import concurrent.futures

def LoadOneFile(name_file:str, key_in_file:str, whether_use_filter:bool):
    load_into_evts = True
    reason_continue = ""
    with np.load(name_file, allow_pickle=True) as f:
        evts_load = f[key_in_file].item()

        ############# Filter abnormal file##########
        # For saving time we can turn off this section
        if whether_use_filter:
            if not evts_load:
                load_into_evts = False
                reason_continue = "Because load_into_evts is null"
            else:
                length_evts_0 = len(list(evts_load.items())[0][1])
                for key in evts_load.keys():
                    if not list(evts_load[key]) :
                        load_into_evts = False
                        reason_continue = f"Because one of itemin evts_load is null ( {key} ) "
                    else:
                        # print(length_evts_0, len(evts_load[key]))
                        if length_evts_0 != len(evts_load[key]):
                            load_into_evts = False
                            reason_continue = "Because length of items in evts_load are not the same"
        ##############################################

        if load_into_evts:
            return evts_load
        else:
            print(f"{name_file} will be continue, {reason_continue}")

def LoadMultiFilesMultiProcess(name_files:str="predict_*.npz", key_in_file:str="dir_events",n_files_to_load=-1,
                   whether_use_filter=False ):
    files_list = glob.glob(name_files)
    f = np.load(files_list[0],
                allow_pickle=True)
    print("key in files:\t",f.files)
    evts_0 = f[key_in_file].item()
    print("Loaded Data Keys:\t", evts_0.keys())
    evts = {}
    for key in evts_0.keys():
        evts[key] = []
    if n_files_to_load!=-1:
        files_list = files_list[1:n_files_to_load]
    else:
        files_list = files_list[1:]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        evts_load = executor.map(LoadOneFile, files_list, [key_in_file]*len(files_list), [whether_use_filter]*len(files_list))
    for evt_load in evts_load:
        for key in evt_load.keys():
            evts[key].extend(evt_load[key])

    for key in evts.keys():
        try:
            evts[key] = np.array(evts[key])
        except Exception:
            print("Something go wrong, continue!")
            continue
    return evts

def LoadOneFileUproot(name_file:str, name_branch:str, list_branch_filter:list=None, return_list:bool=True):
    dir_event = {}
    with up.open(name_file) as f:
        tree = f[name_branch]
        for key in tree.keys():
            if (not list_branch_filter is None) and (key in list_branch_filter):
                continue
            if return_list:
                dir_event[key] = list(np.array(tree[key]))
            else:
                dir_event[key] = np.array(tree[key])
    return dir_event


def LoadMultiROOTFiles(name_files:str="*.root",  name_branch:str="evt", list_branch_filter:list=None, n_files_to_load=-1):
    """
    this function is to load root files with uproot, because uproot.lazy cannot close files correctly,
    so here we use uproot.open() to enable us can load a large amount of files
    :param name_files:
    :param name_branch:
    :param n_files_to_load:
    :return:
    """
    files_list = glob.glob(name_files)
    n_files = len(files_list)
    if n_files == 0:
        print(f"Cannot find related files in {files_list}!!!!!!!!")
        exit(1)
    dir_events = LoadOneFileUproot(files_list[0], name_branch, list_branch_filter, return_list=(n_files != 1))
    if n_files == 1:
        return dir_events
    if n_files_to_load!=-1:
        files_list = files_list[1:n_files_to_load]
    else:
        files_list = files_list[1:]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        evts_load = executor.map(LoadOneFileUproot, files_list, [name_branch]*len(files_list), [list_branch_filter]*len(files_list))
    for evt_load in evts_load:
        for key in evt_load.keys():
            dir_events[key].extend(evt_load[key])
    for key in dir_events.keys():
        try:
            dir_events[key] = np.array(dir_events[key])
        except Exception:
            print(f"Something go wrong in key({key}), continue!")
            continue
    return dir_events

def LoadFileListUprootOptimized(list_files,list_corresponding_keys, name_branch, list_branch_filter:list=None,v_is_one_file=None,
                                use_multiprocess=False):
    """

    :param list_files: files list for files to load, example: ["1.root", "*.root","[1-4].root"]
    :param v_is_one_file:  list or np.ndarray to specify which name file in list_files is single file,
                            so we can put those file together to load using multiprocessing method
    :param list_corresponding_keys: to mark corresponding key in list_files, so we can return dict with those keys
    :param return_shuffle_index: if True , np.ndarray will be return to mark how list_files be shuffle
    :return: dict about different keys' dict_events
    """
    dir_return_diff_file = {}
    symbol_multi_files = ["*", "[","]"]
    list_files = np.array(list_files)
    list_corresponding_keys = np.array(list_corresponding_keys)

    if v_is_one_file is None:
        v_is_one_file = np.array([False if any(symbol in name_file for symbol in symbol_multi_files) else True for name_file in list_files])
    else:
        v_is_one_file = np.array(v_is_one_file)
    # ----------for single file loading ( using multiprocessing to load those files) ---------------------
    n_single_file = Counter(v_is_one_file)[True]
    print("Single File List:\t", list_files[v_is_one_file])

    # -------------For duplicate key, we need to return list so that we can adjoin the dict_events-----
    counter_key = Counter(list_corresponding_keys)
    v_duplicate_keys = [key for key,count in counter_key.items() if count >1]
    v_whether_return_list = np.array([False]*n_single_file)
    for key in v_duplicate_keys:
        v_whether_return_list = v_whether_return_list | (list_corresponding_keys==key)
    #------------------------------------------------------
    if use_multiprocess:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            evts_load = executor.map(LoadOneFileUproot, list_files[v_is_one_file], [name_branch]*n_single_file, [list_branch_filter]*n_single_file, v_whether_return_list)
    else:
        evts_load = []
        for i in range(n_single_file):
            evts_load.append(LoadOneFileUproot(list_files[v_is_one_file][i], name_branch=name_branch, list_branch_filter=list_branch_filter,
                                               return_list=v_whether_return_list[i]))

    for evt_load, key_in_dict,name_file in zip(evts_load, list_corresponding_keys[v_is_one_file],list_files[v_is_one_file]):
        # ----- we adhere the dict_events with the same corresponding keys -----------
        if key_in_dict in dir_return_diff_file:
            for key in dir_return_diff_file[key_in_dict].keys():
                dir_return_diff_file[key_in_dict][key].extend(evt_load[key])
        else:
            dir_return_diff_file[key_in_dict] = evt_load
    for key_in_dict in v_duplicate_keys:
        for key in dir_return_diff_file[key_in_dict].keys():
            dir_return_diff_file[key_in_dict][key] = np.array(dir_return_diff_file[key_in_dict][key])

    #-----------------------------------------------------------------------------------------------------

    #----------- for multi-files loading -----------------------------------------------------------------
    v_is_multi_files = (v_is_one_file == False)
    print("Multi-Files List:\t", list_files[v_is_multi_files])
    print("--------> Loading Multi-files <------------")
    for key_in_dict,name_file_multi_files in tqdm.tqdm(zip(list_corresponding_keys[v_is_multi_files],list_files[v_is_multi_files])):
        dir_return_diff_file[key_in_dict] = LoadMultiROOTFiles(name_file_multi_files, name_branch=name_branch, list_branch_filter=list_branch_filter)

    return {key:dir_return_diff_file[key] for key in list_corresponding_keys}
    # if return_shuffle_index:
    #     index_orignal = np.arange(len(list_files))
    #     index_return = np.concatenate((index_orignal[v_is_one_file],index_orignal[v_is_multi_files]))
    #     return (dir_return_diff_file,index_return)
    #
    # return dir_return_diff_file

if __name__ == "__main__":
    import time
    # t0 = time.perf_counter()
    # evt_original = LoadMultiFiles("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/predict_withpdgdep/predict_*.npz")
    # t1 = time.perf_counter()
    # print(f"Time of Original Way:\t{t1-t0:.2f} s")
    # evt_multiprocess = LoadMultiFilesMultiProcess("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/predict_withpdgdep/predict_*.npz")
    # t2 = time.perf_counter()
    # print(f"Time of MultiProcess Way:\t{t2-t1:.2f} s")
    #
    # for i, key in enumerate(evt_original.keys()):
    #     if i == 0:
    #         print("Original:\t",evt_original[key])
    #         print("MultiProcess:\t",evt_multiprocess[key])
    #
    #     print("Original:\t",len(evt_original[key]))
    #     print("MultiProcess:\t",len(evt_multiprocess[key]))

    # dir_events = LoadMultiROOTFiles(name_files="/afs/ihep.ac.cn/users/v/vavilprod0/Pre-Releases/J21v1r0-Pre2/11/ACU/Co60/Co60_0_0_0/calib/user-root/*.root",
    #                                 name_branch="calibevt")
    # print(dir_events)
    # for key in dir_events.keys():
    #     print(dir_events[key].shape)

    list_name_files = ["/afs/ihep.ac.cn/users/v/valprod0/Pre-Releases/J21v1r0-Pre2/11/ACU/Co60/Co60_0_0_0/calib/user-root/*.root",
                       "/afs/ihep.ac.cn/users/v/valprod0/Pre-Releases/J21v1r0-Pre2/11/ACU/Co60/Co60_0_0_12858.2/calib/user-root/user-calib-0.root",
                       "/afs/ihep.ac.cn/users/v/valprod0/Pre-Releases/J21v1r0-Pre2/11/ACU/Co60/Co60_0_0_-15740.6/calib/user-root/user-calib-[0-9].root"]

    evt_dir = LoadFileListUprootOptimized(list_files=list_name_files, list_corresponding_keys=["1","2","3"],name_branch="calibevt")
    print(evt_dir["1"]["Charge"].shape)
    print(evt_dir["2"]["Charge"].shape)
    print(evt_dir["3"]["Charge"].shape)

