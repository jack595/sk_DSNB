# -*- coding:utf-8 -*-
# @Time: 2022/2/20 19:41
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FitDiffBeamEInit.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from RooFitTools import FitFromArray
import ROOT
def FitBeamE():
    plt.figure()
    plt.plot(range(6))
    plt.show()
    v_mean_out = []
    v_sigma_out = []
    path_save_pdf = "./fit_BeamE_out.pdf"
    c_pdf = ROOT.TCanvas()
    c_pdf.Print(f"{path_save_pdf}[")
    for name_ion, v_BeamE_out in dir_BeamE_out.items():
        v_BeamE_out = np.nan_to_num(v_BeamE_out)
        if name_ion == "Ne_20":
            (mean,sigma) = FitFromArray(v_BeamE_out[ (v_BeamE_out>200) & (v_BeamE_out<1600)],canvas=c_pdf, title=name_ion,std_get_range=False,x_range=(1350,1450))
        else:
            (mean,sigma) = FitFromArray(v_BeamE_out[v_BeamE_out>200],canvas=c_pdf, title=name_ion,std_get_range=False)
        v_mean_out.append(mean)
        v_sigma_out.append(sigma)
        c_pdf.Print(path_save_pdf)

    c_pdf.Print(f"{path_save_pdf}]")
    c_pdf.Close()
    return v_mean_out, v_sigma_out
if __name__ == "__main__":
    with np.load("BeamEnergySmear.npz", allow_pickle=True) as f:
        print(f.files)
        dir_BeamE_init = f["dir_BeamE_init"].item()
        dir_BeamE_out  = f["dir_BeamE_out"].item()


    v_mean_out, v_sigma_out = FitBeamE()
    plt.figure()
    print("Begin")
    plt.errorbar(range(len(v_mean_out)), v_mean_out, yerr=v_sigma_out,fmt='o', markersize=3, capsize=3,
                                    linewidth=1,c="black")
    print("End")
    plt.xticks(range(len(v_mean_out)),list(dir_BeamE_out.keys()))
    plt.ylabel("Kinetic Energy [ MeV ]")
    plt.savefig("./figure/ErrorBarPlot_BeamE_out.png")

    
