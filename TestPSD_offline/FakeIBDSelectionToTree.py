# -*- coding:utf-8 -*-
# @Time: 2021/11/9 15:36
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FakeIBDSelectionToTree.py
# import matplotlib.pylab as plt
# plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import numpy as np

import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from RooFitTools import SaveDictToTFile
from LoadMultiFiles import LoadFileListUprootOptimized

class FakeIBDSelection:
    """
    The class aims to do fake IBD selection by matching the same vertex to filter the prompt signals
    """
    def __init__(self):
        self.dir_events = {}
        self.dir_events_to_save = {}

    def LoadRawData(self, list_files, key_in_dict:str):
        self.dir_events = LoadFileListUprootOptimized(list_files, [key_in_dict]*len(list_files), "evt")[key_in_dict]
        self.dir_events_rec = LoadFileListUprootOptimized(list_files, [key_in_dict]*len(list_files), "PSDUser")[key_in_dict]
        for key in self.dir_events.keys():
            self.dir_events_to_save[key] = []
        print("Keys in events before IBD Selection:\t", self.dir_events.keys())

    def LoadDataAfterIBDSelection(self, list_files, key_in_dict:str):
        self.dir_events_after_IBDSelection = LoadFileListUprootOptimized(list_files, [key_in_dict]*len(list_files),
                                                                         "psdtree")[key_in_dict]
        print("Keys in events after IBD Selection:\t", self.dir_events_after_IBDSelection.keys())


    def MatchVertexToSelectEvents(self):
        print("---------- Check Match Events Pairs -------------")
        i_print = 0
        key_Erec = "recE"

        for r3_tag,Erec in zip(self.dir_events_after_IBDSelection["r3_tag"], self.dir_events_after_IBDSelection["Erec_o"]):
            index_IBDSelection = ( self.dir_events["r3_tag"]== r3_tag) & (self.dir_events_rec[key_Erec]==Erec )
            for key in self.dir_events.keys():
                self.dir_events_to_save[key].extend(self.dir_events[key][index_IBDSelection])

            if i_print < 15:
                print("After Selection:\tR3:\t",r3_tag,"Erec:\t", Erec)
                print("Before Selecting:\tR3:\t", self.dir_events["r3_tag"][index_IBDSelection], "\tErec:\t" ,self.dir_events_rec["recE"][index_IBDSelection])
        print("---------- End of Checking ----------------------")

    def SaveSelectedEvents(self, i_output:int):
        path_save = "root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/TestPSDTool_FakeSelection/"
        SaveDictToTFile(self.dir_events_to_save, f"evt_selected", f"{path_save}atm_selected_{i_output}.root")

    @staticmethod
    def GenScripts( n_total_jobs):
        name_jobs_scripts = "jobs_fakeIBDSelection.sh"
        name_sub_srcript = "sub_fakeIBDSelection.sh"
        list_evtID = np.arange(1, n_total_jobs+2,20)
        template_jobs_scripts = \
"""#/bin/bash
source /afs/ihep.ac.cn/users/l/luoxj/junofs_500G/miniconda3/etc/profile.d/conda.sh && conda activate tf &&
python FakeIBDSelectionToTree.py -s $1 -e $2 -i $3
"""
        template_submit_script =\
"""
hep_sub {} -argu {} {} {}
"""
        with open(name_jobs_scripts, "w") as f:
            f.write(template_jobs_scripts)

        with open(name_sub_srcript, "w") as f:
            f.write(f"chmod 755 {name_jobs_scripts}")

        with open(name_sub_srcript, "a") as f:
            for i in range(len(list_evtID)-1):
                f.write(template_submit_script.format(name_jobs_scripts, list_evtID[i], list_evtID[i+1], i))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("FakeIBDSelection")
    parser.add_argument("--start", "-s", default=1, type=int, help="The start index for root file")
    parser.add_argument("--end"  , "-e", default=21, type=int,help="The end index for root file")
    parser.add_argument("--i_output", "-i", default=1, type=int, help="index for output file")
    parser.add_argument("--n_total_jobs", "-n", default=8000, type=int, help="total number of files")
    parser.add_argument("--gen_scripts", "-g",action="store_true", help="whether generate submit scripts", default=False )
    args = parser.parse_args()

    if args.gen_scripts:
        FakeIBDSelection.GenScripts(args.n_total_jobs)
    else:
        list_files = [f"root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/TestPSDTool/user_atm_{i}.root" for i in range(args.start,args.end)]
        list_files_IBDSelection = [f"/afs/ihep.ac.cn/users/l/luoxj/TestPSD_offline/jobs_ana/root/atmfit_{i}.root" for i in range(args.start,args.end)]
        fake_selection = FakeIBDSelection()
        fake_selection.LoadRawData(list_files, "atm")
        fake_selection.LoadDataAfterIBDSelection(list_files_IBDSelection, "atm")
        fake_selection.MatchVertexToSelectEvents()
        fake_selection.SaveSelectedEvents(args.i_output)




