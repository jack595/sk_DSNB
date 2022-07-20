# -*- coding:utf-8 -*-
# @Time: 2022/6/21 22:27
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: TurnPKLToFitROOTFile.py
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tqdm
import random


# plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
from GlobalFunction import GlobalVal, tag_parentTypeByInt, PreprocessThetaAndBeamXZ
from HistTools import GetBinCenter
from GetPhysicsProperty import NameToPDGID
import ROOT
from LoadMultiFiles import LoadMultiFilesDataframe

def TurnPKLToFitROOTFile(path_files:str, path_save:str, theta_save:int, subtractT0=True, addTSmear=False):
    v_particles = GlobalVal.v_particles
    df_time, dict_replace_chamberID = PreprocessThetaAndBeamXZ( LoadMultiFilesDataframe(path_files,
                                                                                        dict_condition={"chamberID":[5,7]}) )
    print("dict_replace_chamberID:\t",dict_replace_chamberID)
    dir_save = {"time":[], "source":[], "N_nuclei":[]}

    for particle in tqdm.tqdm(v_particles):
        df_tmp = df_time[(df_time["ion"]==particle)&(df_time["mean_theta_int"]==theta_save)]
        h = np.histogram(df_tmp["time"], bins=np.linspace(-10, 30,500))

        df_tmp["source"] = df_tmp["parentPDGID"].apply(lambda pdgID: tag_parentTypeByInt(pdgID, NameToPDGID(particle)))

        if subtractT0:
            dir_save["time"] += list( np.array(df_tmp["time"])-GetBinCenter(h[1])[np.argmax(h[0])] )
        else:
            dir_save["time"] += list(np.array(df_tmp["time"]) )



        dir_save["N_nuclei"] += [int(particle.split("_")[1])]*len(df_tmp["mean_theta_int"])
        dir_save["source"] += list( np.array(df_tmp["source"]) )

    for key in dir_save.keys():
        dir_save[key] = np.array(dir_save[key])
    if addTSmear:
        dir_save["time"] += np.array( [ random.gauss(0, 1) for _ in range(len(dir_save["time"])) ] )

    rdf = ROOT.RDF.MakeNumpyDataFrame(dir_save)
    rdf.Snapshot("Data", path_save )
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Turn Simulation ROOT file in Dataframe")
    parser.add_argument("--L-LS", "-L",type=str)
    parser.add_argument("--NotSubtractT0", action="store_true", help="", default=False)
    parser.add_argument("--AddTSmear", action="store_true", help="", default=False)
    parser.add_argument("--path-input", type=str, help="input pkl files with wildcard",
                        default="/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/OneShotSimulation/pkl_diff_L_LS/PMT_far_*_LS_{L_LS}mm.pkl")
    parser.add_argument("--path-output", type=str, help="output root files",
                        default="/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/OneShotSimulation/root_to_fit/time_particle_theta{theta_save}_LS_{L_LS}mm{option}.root")

    args = parser.parse_args()
    theta_save = 15
    # L_LS = "10"
    L_LS = args.L_LS
    subtractT0 = (not args.NotSubtractT0)
    option = ("" if subtractT0 else "_NotSubtractT0") + ("_AddTSmear" if args.AddTSmear else "")
    print("option:\t", option)
    if subtractT0:
        print("Saving results subtracting t0!!")
    else:
        print("Saving results without subtracting t0!!")


    TurnPKLToFitROOTFile(args.path_input.format(L_LS=L_LS),
                         args.path_output.format(theta_save=theta_save, L_LS=L_LS, option=option),
                         theta_save, subtractT0=subtractT0, addTSmear=args.AddTSmear)

