# -*- coding:utf-8 -*-
# @Time: 2022/6/22 22:39
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: dEdxSpan.py
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from GlobalFunction import GlobalVal,  PreprocessThetaAndBeamXZ
from GetPhysicsProperty import NameToPDGID
from LoadMultiFiles import LoadMultiFilesDataframe

def Get_dEdxSpan(path_files, path_save, chamberID=0):
    v_particles = GlobalVal.v_particles
    df_time, dict_replace_chamberID = PreprocessThetaAndBeamXZ( LoadMultiFilesDataframe(path_files,
                                                                                    dict_condition={"chamberID":[chamberID]}) )
    dict_main_track = {}
    for particle in v_particles:
        dict_main_track[particle] =  NameToPDGID(particle)

    df_time_main_track = pd.DataFrame()
    for particle in v_particles:
        df_time_main_track = pd.concat( (df_time_main_track, df_time[(df_time["ion"]==particle) &
                                                                     (df_time["parentPDGID"]==dict_main_track[particle])]))

    from lmfit.models import GaussianModel
    from HistTools import GetBinCenter
    bins_dEdx = np.concatenate((np.linspace(0,5,300), np.linspace(5,60, 1000)))
    sns.histplot(data=df_time_main_track, x="dE/dx", hue="ion", bins=bins_dEdx, palette="bright",
                 hue_order=v_particles)
    v_colors = sns.color_palette("bright")[:len(v_particles)]
    dict_dEdx_range = {}
    for i,particle in enumerate(v_particles):
        mod = GaussianModel()
        hist = np.histogram(df_time_main_track[df_time_main_track["ion"]==particle]["dE/dx"], bins=bins_dEdx)

        pars = mod.guess(hist[0], x=GetBinCenter(hist[1]))
        out = mod.fit(hist[0], pars, x=GetBinCenter(hist[1]))
        # print(out.fit_report(min_correl=0.25))
        plt.plot( GetBinCenter(hist[1]), out.best_fit,color=v_colors[i], linewidth=1)
        center = out.params["center"].value
        fwhm = out.params["fwhm"].value
        sigma = out.params["sigma"].value
        dict_dEdx_range[particle] = (center-sigma, center+sigma)
        plt.axvspan(dict_dEdx_range[particle][0], dict_dEdx_range[particle][1] ,facecolor=v_colors[i], alpha=0.2)
    plt.xlabel("dE/dx [ MeV/mm ]")
    plt.title("dE/dx Cut (1$\sigma$)")
    
    np.savez(path_save, dict_dEdx_span=dict_dEdx_range)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Turn Simulation ROOT file in Dataframe")
    parser.add_argument("--L-LS", "-L",type=str)

    args = parser.parse_args()
    L_LS = args.L_LS

    Get_dEdxSpan(f"/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/OneShotSimulation/pkl_diff_L_LS/PMT_far_*_LS_{L_LS}mm.pkl",
                 f"/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/OneShotSimulation/L_LS_fitResults/dEdx_Span_{L_LS}mm.npz")
    # plt.show()

