# -*- coding:utf-8 -*-
# @Time: 2022/6/21 21:24
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GlobalFunction.py
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

class GlobalVal:
    bins_time_profile = np.array([-20, -17, -14, -11, -8, -5, -2, 1, 4, 7, 10, 13, 16, 19, 22, 28, 34, 40, 46, 52, 58,
                                  64, 72, 80, 90, 102, 116, 132, 150, 170, 192, 216, 242, 270, 300, 332, 366, 402, 440,
                                  480, 522, 566, 612, 660, 710, 762,816, 872,930,990])+19
    v_particles = ["H_2","He_4",   "Li_6", "B_10", "C_12", "N_14","O_16", "F_18", "Ne_20", "Na_22"]

    dict_replace_sourceTag = {"total":-1,"main track":0, "electron":1,"others":2, "optical photon":3,"gamma":4, "positron":5}
    dict_replace_Num2Tag = {Num:Tag for Tag, Num in dict_replace_sourceTag.items()}
    v_name_timing_constant = ["N1", "tau1", "N2", "tau2", "N3", "tau3"]


def PreprocessThetaAndBeamXZ(df_time:pd.DataFrame):
    dict_replace_chamberID = {}
    dict_replace_chamberID_int = {}
    df_to_extract_theta = df_time[:10000]
    for chamberID in set(df_to_extract_theta["chamberID"]):
        theta_mean = np.round( np.mean(df_to_extract_theta[df_to_extract_theta["chamberID"]==chamberID]["theta"]),0)
        dict_replace_chamberID[chamberID] = f"{theta_mean:.0f} deg"
        dict_replace_chamberID_int[chamberID] = int(theta_mean)


    df_time["mean_theta"] = df_time["chamberID"].replace( dict_replace_chamberID )
    df_time["mean_theta_int"] = df_time["chamberID"].replace( dict_replace_chamberID_int )

    # Binning BeamX
    # df_time.drop(["bin_BeamX", "bin_BeamZ"], axis=1)
    # bins_BeamXZ = np.arange(-25,25, 10)
    n_bins = 5
    bins_BeamXZ = np.linspace(-25,25, n_bins+1)
    df_time["bin_BeamX"] = pd.cut( df_time["BeamX"], bins_BeamXZ)
    df_time["bin_BeamZ"] = pd.cut( df_time["BeamZ"], bins_BeamXZ)
    df_time["num_BeamZ"] = pd.cut( df_time["BeamZ"], bins_BeamXZ, labels=range(n_bins))
    return df_time, dict_replace_chamberID

def tag_parentType(pdgID, pdgID_main_track):
    # if pdgID in [22, 11,-11]:
    #     return "electron"
    if pdgID == 11:
        return "electron"
    elif pdgID == 22:
        return "gamma"
    elif pdgID == -11:
        return "positron"
    elif pdgID == 20022:
        return "optical photon"
    elif pdgID == pdgID_main_track:
        return "main track"
    else:
        return "others"

def tag_parentTypeByInt(pdgID, pdgID_main_track):
    if pdgID == 11:
        return 1
    elif pdgID == 22:
        return 4
    elif pdgID == -11:
        return 5
    elif pdgID == 20022:
        return 3
    elif pdgID == pdgID_main_track:
        return 0
    else:
        return 2

def SortParameters(df_pars:pd.DataFrame):
    for index, row in df_pars.iterrows():
        row_tau = np.array(row[[f"tau{i}" for i in range(1, 4)]])
        row_tau_error = np.array(row[[f"tau{i}_error" for i in range(1, 4)]])
        row_N   = np.array(row[[f"N{i}" for i in range(1, 4)]])
        row_N_error   = np.array(row[[f"N{i}_error" for i in range(1, 4)]])
        if any(np.diff(row_tau)<0):
            v_tau = []
            v_N = []
            v_tau_error = []
            v_N_error = []
            for tau, N, tau_error, N_error in sorted(zip(row_tau, row_N, row_tau_error, row_N_error)):
                v_tau.append(tau)
                v_N.append(N)
                v_tau_error.append(tau_error)
                v_N_error.append(N_error)
            for i,(tau,N, tau_error, N_error) in enumerate(zip(v_tau, v_N, v_tau_error, v_N_error)):
                df_pars.at[ index, f"tau{i+1}"] = tau
                df_pars.at[ index, f"tau{i+1}_error"] = tau_error
                df_pars.at[ index, f"N{i+1}"] = N
                df_pars.at[ index, f"N{i+1}_error"] = N_error
    return df_pars

def PreprocessDF_pars(df_pars:pd.DataFrame):
    dict_N2dEdx = {}
    with np.load("/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/run/BeamEnergy.npz",allow_pickle=True) as f:
        for name_ion, dE_dx in zip(f["name"],f["dE_dx"]):
            dict_N2dEdx[int(name_ion.split("_")[1])] = dE_dx
    df_pars["dE/dx"] = df_pars["N_nuclei"].replace(dict_N2dEdx)
    return df_pars

def SetdEdxForDataframePars(df_pars:pd.DataFrame):
    v_L_LS = ["0.5", "1.0", "2", "5", "7", "10"]
    df_pars = PreprocessDF_pars(df_pars)
    df_pars = SortParameters(df_pars)

    dict_dEdx_to_df = {"ion":[], "L_LS":[], "dE/dx":[], "sigma(dE/dx)":[], "N_nuclei":[]}
    for L_LS in v_L_LS:
        with np.load(f"/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/OneShotSimulation/L_LS_fitResults/dEdx_Span_{L_LS}mm.npz", allow_pickle=True) as f:
            dict_dEdxSpan = f["dict_dEdx_span"].item()
        for particle, v_range in dict_dEdxSpan.items():
            dict_dEdx_to_df["ion"].append(particle)
            dict_dEdx_to_df["L_LS"].append(L_LS)
            dict_dEdx_to_df["dE/dx"].append(np.mean(v_range))
            dict_dEdx_to_df["sigma(dE/dx)"].append(v_range[1]-np.mean(v_range))
            dict_dEdx_to_df["N_nuclei"].append(int(particle.split("_")[1]))

    df_dEdx = pd.DataFrame.from_dict(dict_dEdx_to_df).set_index(["N_nuclei", "L_LS"])

    v_dEdx_map = []
    v_dEdx_sigma_map = []
    for index, row in df_pars.iterrows():
        df_tmp = df_dEdx.loc[(row["N_nuclei"], row["L_LS"])]
        v_dEdx_map.append(df_tmp["dE/dx"])
        v_dEdx_sigma_map.append(df_tmp["sigma(dE/dx)"])
    df_pars["dE/dx"] = v_dEdx_map
    df_pars["sigma(dE/dx)"] = v_dEdx_sigma_map
    return df_pars

def ConvertRecursiveFraction(df_pars:pd.DataFrame):
    if "f2" not in df_pars.columns:
        df_pars["f2"] = df_pars["N2"]
        df_pars["f3"] = df_pars["N3"]
        df_pars["f2_error"] = df_pars["N2_error"]
    df_pars["N2"] = (1-df_pars["N1"])*df_pars["f2"]
    df_pars["N3"] = (1-df_pars["N1"])*(1-df_pars["f2"])
    df_pars["N2_error"] = np.sqrt( ( df_pars["f2"]*df_pars["N1_error"] )**2 +
                                   ( (1-df_pars["N1"])*df_pars["N2_error"]  )**2 )
    df_pars["N3_error"] = np.sqrt( ( (1-df_pars["f2"])*df_pars["N1_error"] )**2 +
                                   ( (1-df_pars["N1"])*df_pars["N2_error"]  )**2 )
    return df_pars

def PlotTimeConstantModel(axes=None, xlim=(0, 65)):
    if axes is None:
        fig, axes = plt.subplots(6,1,sharex="col",figsize=(8,9))
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    dir_v_timing_constant = {}

    dir_mean_dE_dx_with_quench = {'alpha': 122.773224, 'Co60': 0.6747912, 'AmC': 29.574236}
    dir_v_timing_constant[dir_mean_dE_dx_with_quench["Co60"]] = [79.9,4.93,17.1,20.6,3,190]
    dir_v_timing_constant[dir_mean_dE_dx_with_quench["AmC"]] = [65,4.93,23.1,34,11.9,220]
    dir_v_timing_constant[dir_mean_dE_dx_with_quench["alpha"]] = [65,4.93,22.8,35,12.2,220]

    for particle in dir_v_timing_constant.keys():
        for i in range(len(GlobalVal.v_name_timing_constant)):
            if "N" in GlobalVal.v_name_timing_constant[i]:
                dir_v_timing_constant[particle][i] = dir_v_timing_constant[particle][i]*0.01
    df_model = pd.DataFrame.from_dict(dir_v_timing_constant, orient="index", columns=GlobalVal.v_name_timing_constant)

    for i_constant, name_time_constant in enumerate(GlobalVal.v_name_timing_constant):
        v_x = np.array(df_model.index)
        v_y = np.array(df_model[name_time_constant] )

        axes[i_constant].plot(v_x, v_y, color=colors[i_constant], label=name_time_constant)
        axes[i_constant].scatter(v_x,v_y, color=colors[i_constant])
        axes[i_constant].set_xlim(xlim[0], xlim[1])
        axes[i_constant].legend(loc="right")
    axes[-1].set_xlabel("dE/dx [ MeV/mm ]")
    return axes

def PlotTimeConstantModelBlackSkin(axes=None, xlim=(0, 55)):
    if axes is None:
        fig, axes = plt.subplots(6,1,sharex="col",figsize=(8,9))
    colors = ["black"]*10
    dir_v_timing_constant = {}

    dir_mean_dE_dx_with_quench = {'alpha': 122.773224, 'Co60': 0.6747912, 'AmC': 29.574236}
    dir_v_timing_constant[dir_mean_dE_dx_with_quench["Co60"]] = [79.9,4.93,17.1,20.6,3,190]
    dir_v_timing_constant[dir_mean_dE_dx_with_quench["AmC"]] = [65,4.93,23.1,34,11.9,220]
    dir_v_timing_constant[dir_mean_dE_dx_with_quench["alpha"]] = [65,4.93,22.8,35,12.2,220]

    for particle in dir_v_timing_constant.keys():
        for i in range(len(GlobalVal.v_name_timing_constant)):
            if "N" in GlobalVal.v_name_timing_constant[i]:
                dir_v_timing_constant[particle][i] = dir_v_timing_constant[particle][i]*0.01
    df_model = pd.DataFrame.from_dict(dir_v_timing_constant, orient="index", columns=GlobalVal.v_name_timing_constant)

    for i_constant, name_time_constant in enumerate(GlobalVal.v_name_timing_constant):
        v_x = np.array(df_model.index)
        v_y = np.array(df_model[name_time_constant] )

        axes[i_constant].plot(v_x, v_y, color=colors[i_constant])
        axes[i_constant].set_ylabel(name_time_constant)
        axes[i_constant].scatter(v_x,v_y, color=colors[i_constant])
        axes[i_constant].set_xlim(xlim[0], xlim[1])
        # axes[i_constant].legend(loc="right")
    axes[-1].set_xlabel("dE/dx [ MeV/mm ]")
    return axes

def Plot_df_pars(df_pars:pd.DataFrame, axes=None, j=0, source=0, dict_ylim:dict=None,
                 AddLegend=True,dict_colors=None,  label="", loc_legend=None):
    import seaborn as sns
    if dict_colors is None:
        dict_colors = {-1:"black", 0:"olivedrab", 1:"darkorange"}
    if label == "":
        label=GlobalVal.dict_replace_Num2Tag[source]
    if axes is None:
        fig, axes = plt.subplots(6,1,sharex="col",figsize=(8,9))
    for i_constant, name_time_constant in enumerate(GlobalVal.v_name_timing_constant):
        axes[i_constant].errorbar(df_pars["dE/dx"], df_pars[name_time_constant],
                                 yerr=df_pars[name_time_constant+"_error"],xerr=df_pars["sigma(dE/dx)"],
                                 color=dict_colors[source], ls='none', marker="o", capsize=5, capthick=1,
                                 ecolor=dict_colors[source],
                                 markersize=5,label=label if (i_constant==j)  else "")
        if AddLegend:
            axes[i_constant].legend(loc=loc_legend)
        if not dict_ylim is None:
            axes[i_constant].set_ylim(dict_ylim[name_time_constant][0],dict_ylim[name_time_constant][1] )

def MarkSideOutOfLS(df_time:pd.DataFrame, X_border:float=24.99999999, Y_border:float=0.999999999, Z_border:float=24.99999999):
    dict_NumMapSide = {1:'top', 2:'bottom', 3:'forward', 4:'backward', 5:'left', 6:'right'}
    v_outLS_side = np.zeros(len(df_time))
    v_outLS_side[df_time["YGoOutLS"]>= Y_border] = 1
    v_outLS_side[df_time["YGoOutLS"]<=-Y_border] = 2
    v_outLS_side[df_time["XGoOutLS"]>= X_border] = 4
    v_outLS_side[df_time["XGoOutLS"]<=-X_border] = 3
    v_outLS_side[df_time["ZGoOutLS"]>= Z_border] = 5
    v_outLS_side[df_time["ZGoOutLS"]<=-Z_border] = 6

    from NumpyTools import Replace
    df_time["SideOutLS"] = Replace(v_outLS_side, dict_NumMapSide, else_item="Cherekov")
    return df_time
