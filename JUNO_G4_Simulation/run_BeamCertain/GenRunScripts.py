# -*- coding:utf-8 -*-
# @Time: 2022/2/19 15:54
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenRunScripts.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/")
from run.GenRunScripts import GenRunScripts
if __name__ == "__main__":

    import periodictable as pt
    import os

    ###    Generate list of particles to simulate
    list_elements_gen = list(pt.elements)[1:20]
    list_charge = range(1,len(list_elements_gen)+1)
    input_list = []

    from PeriodictableTools import PeriodictableTools
    table_tool = PeriodictableTools()
    v_name_to_gen = ["H_2","He_4",   "Li_6", "B_10", "C_12", "N_14","O_16", "F_18", "Ne_20", "Na_22"]
    for name in v_name_to_gen:
        Z = table_tool.MapToCharge(name.split("_")[0])
        N = name.split("_")[1]
        input_list.append(f"{Z} {N} {Z} 0")


    f = np.load("/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/run/BeamEnergy.npz",allow_pickle=True)
    dir_ion = \
        {
            "name": v_name_to_gen,
            "input_list": input_list,
            "BeamE[MeV/u]":f["BeamE"]
        }

    path_run = "run_BeamCertain"
    gen_scripts = GenRunScripts()
    template_outfile = gen_scripts.work_path+f"/{path_run}/"+"mac/{}.mac"
    template_outfile_root = gen_scripts.work_path+f"/{path_run}/"+"root/{}.root"
    template_outfile_job = gen_scripts.work_path+f"/{path_run}/"+"jobs/{}.sh"

    #### Check directories
    from Bash_Tool import CheckDirectory
    CheckDirectory(os.path.dirname(template_outfile))
    CheckDirectory(os.path.dirname(template_outfile_root))
    CheckDirectory(os.path.dirname(template_outfile_job))

    for i in range(len(dir_ion["name"])):
        mac_file = template_outfile.format(dir_ion["name"][i])
        root_file = template_outfile_root.format(dir_ion["name"][i])
        job_file = template_outfile_job.format(dir_ion["name"][i])
        Z = float(dir_ion["input_list"][i].split(" ")[1])
        print(Z, Z*dir_ion["BeamE[MeV/u]"][i])
        gen_scripts.GenMacFileCertainE(mac_file, dir_ion["input_list"][i], Ecertain=Z*dir_ion["BeamE[MeV/u]"][i],nEvts=10000)
        gen_scripts.GenScripts(input_mac_file=mac_file, out_root_file=root_file, name_scripts=job_file, L_LS=5)
        os.system("chmod 755 "+job_file)