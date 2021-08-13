import numpy as np
import glob
def LoadMultiFiles(name_files:str="predict_*.npz", key_in_file:str="dir_events"):
    files_list = glob.glob(name_files)
    f = np.load(files_list[0],
                allow_pickle=True)
    evts_0 = f[key_in_file].item()
    print("Loaded Data Keys:\t", evts_0.keys())
    evts = {}
    for key in evts_0.keys():
        evts[key] = []
    for file in files_list:
        with np.load(file, allow_pickle=True) as f:
            evts_load = f[key_in_file].item()
            for key in evts_load.keys():
                evts[key].extend(evts_load[key])
    for key in evts.keys():
        try:
            evts[key] = np.array(evts[key])
        except Exception:
            print("Something go wrong, continue!")
            continue
    return evts
if __name__ == "__main__":
    print(LoadMultiFiles("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/predict_withpdgdep/predict_*.npz"))
