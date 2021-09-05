# -*- coding:utf-8 -*-
# @Time: 2021/9/5 17:20
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotMichealElectronAndNeutron.py
import matplotlib.pylab as plt
import numpy as np
import uproot4 as up

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

class PlotMichealEAndNeutronTools:
    def __init__(self):
        self.tree_micheal_electron = None
        self.tree_neutron_capture = None
        self.name_file_last = None
        self.tree_evts = None
    def SetDataset(self, name_file:str, key_tree_micheal_e="michael",
                   key_tree_neutron="nCapture", key_evts="evt"):
        if name_file != self.name_file_last:
            with up.open(name_file) as f:
                self.tree_micheal_electron = f[key_tree_micheal_e]
                self.tree_neutron_capture = f[key_tree_neutron]
                self.tree_evts = f[key_evts]
                self.dir_evts_micheal_electron = {}
                self.dir_evts_neutron_capture = {}
                for key in self.tree_micheal_electron.keys():
                    self.dir_evts_micheal_electron[key] = np.array(self.tree_micheal_electron[key])
                for key in self.tree_neutron_capture.keys():
                    self.dir_evts_neutron_capture[key] = np.array(self.tree_neutron_capture)

                self.name_file_last = name_file

                self.dir_evts = {}
                for key in self.tree_evts.keys():
                    self.dir_evts[key] = np.array(self.tree_evts[key])
        else:
            pass
    def PrintMichealElctronVertex(self, entry_source):
        evtID = self.dir_evts["evtID"][entry_source]
        index_evtID = (self.dir_evts_micheal_electron["evtID"] == evtID)
        print(self.dir_evts_micheal_electron["x"][index_evtID])
        print(self.dir_evts_micheal_electron["y"][index_evtID])
        print(self.dir_evts_micheal_electron["z"][index_evtID])



