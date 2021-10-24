import numpy as np
import glob
from copy import copy
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

if __name__ == "__main__":
    print(LoadMultiFiles("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/predict_withpdgdep/predict_*.npz"))
