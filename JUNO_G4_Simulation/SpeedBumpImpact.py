# -*- coding:utf-8 -*-
# @Time: 2022/1/16 10:13
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: SpeedBumpImpact.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")





######################### Draw Ek Distribution #########################################
def DrawEkDistribution(dir_LS_diff_L):
    #%%
    dir_v_max_Ek = {}
    for key in dir_LS_diff_L.keys():
        v_Ek = []
        dir_LS = dir_LS_diff_L[key]
        for v_step_Ek in dir_LS["step_KineticE"]:
            v_Ek.append(np.max(v_step_Ek))
        dir_v_max_Ek[key] = np.array(v_Ek)
    # print(len(v_Ek))
    # print(dir_LS_diff_L["0.5"]["step_KineticE"])
    #%%
    for key in dir_v_max_Ek.keys():
        plt.hist(dir_v_max_Ek[key]/4, bins=100, label=f"{key} cm ( Entries={len(dir_v_max_Ek[key])} )", histtype="step")
    plt.title("Alpha ( Beam Energy = 100 MeV/u )")
    plt.legend(loc="upper left")
    plt.xlabel("Beam Energy [ MeV/u ]")
    plt.savefig("./figure/dE_dx_SpeedBump.png")
    v_L_SpeedBump = []
    v_eff = []
    for key in dir_LS_diff_L.keys():
        v_L_SpeedBump.append(key)
        # print(dir_geninfo_diff_L[key].keys())
        v_eff.append(len(dir_v_max_Ek[key])/len(dir_geninfo_diff_L[key]["E_init"]))
    plt.figure()
    plt.scatter(v_L_SpeedBump, v_eff)
    plt.title("100 MeV/u alpha")
    plt.xlabel("Length Of Aluminum Layer")
    plt.ylabel("Efficiency")
    plt.savefig("./figure/Efficiency_SpeedBump.png")
    plt.show()
    v_Ek_mean_fit = []
    v_Ek_sigma_fit = []
    v_L_SpeedBump = []
    path_save_pdf = "./figure/figfit.pdf"
    c_pdf = ROOT.TCanvas()
    c_pdf.Print(f"{path_save_pdf}[")
    for i,(L_SpeedBump,v_max_Ek) in enumerate(dir_v_max_Ek.items()):
        print(f"c{i}")
        # locals()[f"c{i}"] = ROOT.TCanvas(f"c{i}", f"c{i}",800,600)
        print("Try to Fit from Array")
        (mean, sigma)= FitFromArray(v_max_Ek, xlabel="Beam Energy [ MeV/u ]",
                     x_range=(0, 420),x_range_fit=None,canvas=c_pdf,
                           title=f"{L_SpeedBump} mm")
        c_pdf.Print(path_save_pdf)
        v_Ek_mean_fit.append(mean)
        v_Ek_sigma_fit.append(sigma)
        v_L_SpeedBump.append(L_SpeedBump)
    c_pdf.Print(f"{path_save_pdf}]")
    plt.errorbar(v_L_SpeedBump, np.array(v_Ek_mean_fit)/4, yerr=v_Ek_sigma_fit,fmt='o', markersize=3, capsize=3,
                                    linewidth=1,c="black")
    plt.title("100 MeV/u alpha")
    plt.xlabel("Length Of Aluminum Layer")
    plt.ylabel("Beam Energy [ MeVu ]")
    plt.savefig("./figure/FitResult.png")
    plt.show()

######################################################################################################################

################################### Draw dE/dx Distribution #######################################################
def DrawdEdxDistribution(dir_dE_dx_diff_L):
    plt.figure()
    for key, dir_dE_dx in dir_dE_dx_diff_L.items():
        plt.hist(dir_dE_dx["dE_dx_main_track"],bins=np.linspace(0,30, 300), histtype="step", label=key+" mm")


    plt.legend()
    plt.xlabel("dE/dx [ MeV/mm ]")
    plt.savefig("./figure/dE_dx_distribution.png")
    plt.show()

    path_save_pdf = "./figure/figfit_dE_dx.pdf"
    v_dE_dx_mean = []
    v_dE_dx_sigma = []
    v_L_SpeedBump = []

    c_pdf = ROOT.TCanvas()
    c_pdf.Print(f"{path_save_pdf}[")
    for key, dir_dE_dx in dir_dE_dx_diff_L.items():
        if float(key) < 3.5:
            fit_range_sigma= 0.2
        else:
            fit_range_sigma = 5
        (mean,sigma) = FitFromArray(dir_dE_dx["dE_dx_main_track"], xlabel="dE/dx [ MeV/mm ]",
                                x_range=(0,30), x_range_fit=None,canvas=c_pdf, sigma_range=(0.001,10), fit_range_sigma=fit_range_sigma,
                                title=f"{key} mm",bins=np.linspace(0,30,300))
        v_dE_dx_mean.append(mean)
        v_dE_dx_sigma.append(sigma)
        v_L_SpeedBump.append(float(key))
        c_pdf.Print(path_save_pdf)

    c_pdf.Print(f"{path_save_pdf}]")
    plt.figure()
    plt.errorbar(v_L_SpeedBump, np.array(v_dE_dx_mean), yerr=v_dE_dx_sigma,fmt='o', markersize=3, capsize=3,
                                    linewidth=1,c="black")
    plt.title("100 MeV/u alpha")
    plt.xlabel("Length Of Aluminum Layer")
    plt.ylabel("dE/dx [ MeV/mm ]")
    plt.savefig("./figure/FitResult_dE_dx.png")
    plt.show()

if __name__ == '__main__':
    from RooFitTools import FitFromArray
    import ROOT

    from glob import glob
    path_files = "/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/run_diff_L_SpeedBump/alpha_no_optical_{}mm.root"
    file_list = glob(path_files.format("*"))
    print(file_list)

    from LoadMultiFiles import LoadOneFileUproot

    from FunctionFor_dE_dx import GetDirForNoOpticalAnalyze
    from copy import copy

    dir_LS_diff_L = {}
    dir_geninfo_diff_L = {}
    dir_dE_dx_diff_L = {}
    dir_key_to_float = {}
    for file in file_list:
        key = file.split("_")[-1].split("mm.root")[0]
        key_float  = float(key)
        dir_key_to_float[key] = key_float
    tuple_key_to_float_sorted = sorted(dir_key_to_float.items(), key=lambda x: x[1])
    print(tuple_key_to_float_sorted)

    for key in tuple_key_to_float_sorted:
        key = key[0]
        file = path_files.format(key)
        dir_LS = LoadOneFileUproot(file, name_branch="GdLS_log",
                               return_list=False)
        dir_geninfo = LoadOneFileUproot(file, name_branch="genInfo",
                               return_list=False)
        pdgID_certain,dir_dE_dx =GetDirForNoOpticalAnalyze(dir_LS,dir_geninfo)
        dir_LS_diff_L[key] = copy(dir_LS)
        dir_geninfo_diff_L[key] = copy(dir_geninfo)
        dir_dE_dx_diff_L[key] = copy(dir_dE_dx)

    # DrawEkDistribution(dir_LS_diff_L)
    DrawdEdxDistribution(dir_dE_dx_diff_L)