# -*- coding:utf-8 -*-
# @Time: 2022/6/13 14:09
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: TurnSimRootFileToDF.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/")

from LoadMultiFiles import MultiFilesEvtIDMapProperty
from LoadMultiFiles import LoadOneFileUproot, LoadMultiROOTFilesEOS, LoadMultiFileUprootMultiBranch
from FunctionFor_dE_dx import GetNPE, GetDirForNoOpticalAnalyze
from GetPhysicsProperty import NameToPDGID
from copy import copy
import pandas as pd
import tqdm
from IPython.display import display

from LoadMultiFiles import LoadFileListUprootOptimized, MergeEventsDictionary
def GetPhotonParentInfo(dir_PMT, dir_LS):
    from NumpyTools import Replace
    v2d_parent_pdgID = []
    v2d_parent_dEdx = []
    for i, (fileNo, evtID) in enumerate(zip(dir_PMT["LoadedFileNo"],dir_PMT["evtID"])):
        index_LS_evtID = np.where( (dir_LS["evtID"]==evtID) & (dir_LS["LoadedFileNo"]==fileNo) )[0][0]
        v_parentID = dir_PMT["step_ParentID"][i]
        dict_map_pdgID = dict( set( list(zip( dir_LS["step_trackID"][index_LS_evtID], dir_LS["step_pdgID"][index_LS_evtID]))) )
        dict_map_dEdx = dict( set( list(zip( dir_LS["step_trackID"][index_LS_evtID], dir_LS["step_Edep"][index_LS_evtID]/dir_LS["step_dx"][index_LS_evtID] ))) )
        v2d_parent_pdgID.append( Replace(v_parentID, dict_map_pdgID, else_item=20022) )
        v2d_parent_dEdx.append( Replace(v_parentID, dict_map_dEdx, else_item=0) )
    return v2d_parent_pdgID, v2d_parent_dEdx


def TurnSimRootFileToDF(file_template:str, fileNo_start:int, fileNo_end:int, particle:str):


    # filter_leaves = ['xyz_center', 'step_x', 'step_y', 'step_z' ,'step_Edep', 'step_Equench',"step_dx",
    #                  "step_trackID" ]

    dict_to_df = {"ion":[], "time":[], "chamberID":[], "BeamX":[], "BeamZ":[], "Ek":[], "dE_quench":[],
                  "dE/dx":[], "parentPDGID":[], "parent_dEdx":[], "theta":[], "isCherenkov":[]}
    dict_nEvts = {}

    # dir_geninfo = LoadMultiROOTFilesEOS( file_template, fileNo_start, fileNo_end, name_branch="genInfo", use_multiprocess=False )
    # dir_PMT_far = LoadMultiROOTFilesEOS( file_template, fileNo_start, fileNo_end, name_branch="PMT_log_R7600",  list_branch_filter=filter_leaves, use_multiprocess=False)
    # dir_LS = LoadMultiROOTFilesEOS(  file_template, fileNo_start, fileNo_end, name_branch="GdLS_log", use_multiprocess=False)
    v_files = []
    for fileNo in range(fileNo_start, fileNo_end+1):
        v_files.append(file_template.format(fileNo))
    dict_MultiBranches = LoadMultiFileUprootMultiBranch(v_files, v_name_branch=["genInfo", "PMT_log_R7600", "GdLS_log"], 
                                                        templateToExtractFileNo=file_template.replace("{}", "(.*)"))

    dir_geninfo = copy(dict_MultiBranches["genInfo"])
    dir_PMT_far = copy(dict_MultiBranches["PMT_log_R7600"])
    dir_LS = copy(dict_MultiBranches["GdLS_log"])

    pdgID_certain,dir_dEdx = GetDirForNoOpticalAnalyze(dir_events=dir_LS, dir_geninfo=dir_geninfo, pdgID=NameToPDGID(particle),
                                                       evtIDMap=True)

    # Find Event Information, like beam position and dE/dx
    v_dE_quench = MultiFilesEvtIDMapProperty( dir_PMT_far["evtID"], np.array(dir_PMT_far["LoadedFileNo"],dtype=int),
                                             dir_dEdx["Equench"], dir_dEdx["evtID"], dir_dEdx["fileNo"])

    v_dEdx = MultiFilesEvtIDMapProperty( dir_PMT_far["evtID"], np.array(dir_PMT_far["LoadedFileNo"],dtype=int),
                                              dir_dEdx["dE_dx_main_track"], dir_dEdx["evtID"], dir_dEdx["fileNo"])

    v_BeamXYZ = MultiFilesEvtIDMapProperty( dir_PMT_far["evtID"],np.array(dir_PMT_far["LoadedFileNo"],dtype=int),
                                    dir_geninfo["XYZ"],dir_geninfo["evtID"],
                                    np.array(dir_geninfo["LoadedFileNo"],dtype=int))

    v2d_BeamX = []
    v2d_BeamZ = []
    v2d_dE_quench = []
    v2d_dEdx = []
    for BeamXYZ, v_chamberID,dE_quench, dEdx in  zip(v_BeamXYZ, dir_PMT_far["step_chamberID"],
                                               v_dE_quench,v_dEdx):
        v2d_BeamX.append( [BeamXYZ[0]]*len(v_chamberID))
        v2d_BeamZ.append( [BeamXYZ[2]]*len(v_chamberID))
        v2d_dE_quench.append( [dE_quench]*len(v_chamberID))
        v2d_dEdx.append( [dEdx]*len(v_chamberID))

    dict_nEvts[particle] = len(dir_geninfo["E_init"])
    #############################################################
    # Find photon Parent pdgID
    v2d_parent_pdgID, v2d_parent_dEdx = GetPhotonParentInfo(dir_PMT=dir_PMT_far, dir_LS=dir_LS)
    ##########################################################################

    index_pdgID = np.concatenate(dir_PMT_far["step_pdgID"])==20022
    v_time =  np.concatenate(dir_PMT_far["step_t"])[index_pdgID]
    v_isCherekov = np.array( np.concatenate( dir_PMT_far["step_isCherenkov"])[index_pdgID], dtype=np.int )
    v_Ek = np.concatenate(dir_PMT_far["step_KineticE"])[index_pdgID]
    dict_to_df["time"] += list(v_time)
    dict_to_df["isCherenkov"] += list(v_isCherekov)
    dict_to_df["ion"]  += [particle]*len(v_time)
    dict_to_df["chamberID"] += list( np.concatenate(dir_PMT_far["step_chamberID"])[index_pdgID] )
    dict_to_df["BeamX"] += list( np.concatenate(v2d_BeamX) )
    dict_to_df["BeamZ"] += list( np.concatenate(v2d_BeamZ) )
    dict_to_df["Ek"] += list( v_Ek)
    dict_to_df["dE_quench"] += list( np.concatenate(v2d_dE_quench) )
    dict_to_df["dE/dx"] += list( np.concatenate(v2d_dEdx) )
    dict_to_df["parentPDGID"] += list( np.concatenate(v2d_parent_pdgID) )
    dict_to_df["parent_dEdx"] += list( np.concatenate(v2d_parent_dEdx)[index_pdgID] )


    v_y = np.concatenate(dir_PMT_far["step_y"])[index_pdgID]
    v_z = np.concatenate(dir_PMT_far["step_z"])[index_pdgID]
    dict_to_df["theta"] += list( np.degrees(np.arccos(-v_y/(v_z**2+v_y**2)**0.5)) )

    for key, items in dict_to_df.items():
        print(key, len(items))
    df_time = pd.DataFrame.from_dict(dict_to_df)
    display(df_time)
    display(df_time[df_time["chamberID"]==11])
    display(df_time[df_time["chamberID"]==1])
    return df_time
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Turn Simulation ROOT file in Dataframe")
    parser.add_argument("--particle", "-p",  type=str, help="name of particle", default="He_4")
    parser.add_argument("--directory", "-d",  type=str, help="directory of input_file", default="root2_addEvtIDinGen")
    parser.add_argument("--outfile", "-o",  type=str, help="path of output_file",
                        default="/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/OneShotSimulation/pkl/PMT_far_He_4.pkl")
    parser.add_argument("--fileNo_start", "-s",type=int, default=3001)
    parser.add_argument("--fileNo_end", "-e",type=int, default=3003)

    args = parser.parse_args()

    particle = args.particle
    df = TurnSimRootFileToDF(f"root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/JUNO_G4_Simulation/OneShotSimulation/{args.directory}/"+particle+"_10cm_{}.root",
                        args.fileNo_start,args.fileNo_end, particle)
    df.to_pickle(args.outfile)


