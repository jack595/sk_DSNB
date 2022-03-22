# -*- coding:utf-8 -*-
# @Time: 2022/3/5 10:05
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: RunExtractFeactures.py
import numpy as np

import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/code/")

import argparse
from LoadMultiFiles import LoadOneFileUproot, LoadOneFileUprootCertainEntries
from ExtractFeatureForAfterPulse import ExtractFeature
from ExtractFeaturesToStudyTruth import ExtractFeatureForTruth
import pandas as pd
from DictTools import FilterEventsDict
from NumpyTools import Replace
from CommonVariables import CommonVariables

if __name__ == "__main__":
    option_time_profile = "_NotSubtractTOF"
    # option_time_profile = "_noShift"

    parser = argparse.ArgumentParser(description='ExtractFeatures to Separate AfterPulse')
    parser.add_argument("--template-path-PSDTools", "-p", type=str,
                        default=f"/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/PSD/root{option_time_profile}"+"/user_PSD_{}__SN.root",
                        help="path template of input about PSDTools")
    parser.add_argument("--template-path-evtTruth", "-e", type=str,
                        default="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/tag_event/root/sn_tag_{}.root",
                        help="path template of input about evtTruth")
    parser.add_argument("--fileNo", "-n", type=int, default=0, help="No. of file to execute")
    parser.add_argument("--template-outfile", "-o", type=str, default="try_{}.root", help="name of outfile")
    parser.add_argument("--save-npz", "-s", action="store_true", default=False, help="Whether to save file as npz")
    parser.add_argument("--save-truth", action="store_true", default=False, help="Whether to save file as Time Truth")
    parser.add_argument("--full", action="store_true", default=False, help="Save Full Tags")
    arg = parser.parse_args()

    # Set Variables
    list_truth_to_save = [ 'PulseTimeTruth', "TriggerTime"]
    if arg.full:
        v_tags = None
    else:
        v_tags = list(CommonVariables.map_tag.keys())
    Ecut = None
    t_length_buffer = 1e6

    if arg.save_truth:
        dir_evts = LoadOneFileUprootCertainEntries(arg.template_path_PSDTools.format(arg.fileNo),
                                                   name_branch="evt",
                                                   return_list=False, n_entries_load=10000)
        dir_map = LoadOneFileUprootCertainEntries(arg.template_path_evtTruth.format(arg.fileNo), name_branch='evtTruth',
                                                  list_load_branch=["evtType", "recX",
                                                                    "recY", "recZ","TriggerTimeInterval",
                                                                    "TriggerTime"]+list_truth_to_save,
                                                  return_list=False,n_entries_load=10000)
    else:
        dir_evts = LoadOneFileUproot(arg.template_path_PSDTools.format(arg.fileNo),     name_branch="evt",
                                                   return_list=False)
        dir_map = LoadOneFileUproot(arg.template_path_evtTruth.format(arg.fileNo), name_branch='evtTruth',
                                                  return_list=False)

    bins = np.loadtxt(f"/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/PSD/Bins_Setting{option_time_profile}.txt", delimiter=",")

    if arg.save_truth:
        dir_variables =ExtractFeatureForTruth(dir_map, dir_evts, bins, v_tags, Ecut,
                                                t_length_buffer, list_truth_to_save=list_truth_to_save)
    else:
        if v_tags== None:
            dir_variables,v_tags = ExtractFeature(dir_map, dir_evts, bins, v_tags, Ecut,
                                           t_length_buffer, save_h_time=arg.save_npz, save_truth=arg.save_truth)

            name_file = arg.template_outfile.replace(".root", ".npz").format("")
            np.savez(name_file, dir_variables=dir_variables)
            exit(0)
        else:
            dir_variables =ExtractFeature(dir_map, dir_evts, bins, v_tags, Ecut,
                                  t_length_buffer, save_h_time=arg.save_npz, save_truth=arg.save_truth)


    # Separate different events into different TTrees
    from collections import Counter

    for tree_tag in ["AfterPulse", "eES_pES"]:
        index_tag = [False]*len(dir_variables["tag"])
        for tag in v_tags:
            if tag in tree_tag:
                index_tag = index_tag | (np.array(dir_variables["tag"])==tag)

        dir_variables_with_tag = FilterEventsDict(dir_variables, index_tag)

        print(tree_tag, Counter(dir_variables_with_tag["tag"]))

        # Prepare data to Save tag
        dir_variables_with_tag["tag"] = Replace(dir_variables_with_tag["tag"], CommonVariables.map_tag)

        for key in dir_variables_with_tag.keys():
            if key == "tag":
                dir_variables_with_tag[key] = np.array(dir_variables_with_tag[key], dtype=np.int32)
            elif "Truth" in key:
                continue
            else:
                dir_variables_with_tag[key] = np.array(dir_variables_with_tag[key], dtype=np.float64)


        # df_variables = pd.DataFrame.from_dict(dir_variables_with_tag)
        # print( list(dir_variables_with_tag["tag"]) )

        if arg.save_npz:
            name_file = arg.template_outfile.replace(".root", ".npz").format(tree_tag)
            print(name_file)
            np.savez(name_file, dir_variables=dir_variables_with_tag)
        else:
            # Save results into root file
            import ROOT
            rdf = ROOT.RDF.MakeNumpyDataFrame(dir_variables_with_tag)
            rdf.Snapshot("Features", arg.template_outfile.format(tree_tag))
            rdf.Display().Print()
