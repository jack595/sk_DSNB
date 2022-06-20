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
    template_outfile = "./mac/{}.mac"
    template_mac_file = \
"""/control/verbose 1
/run/verbose 1
/event/verbose 0
/tracking/verbose 0

# analysis manager
#/analysis/setFileName test.root

## Plane mode ##
# /gps/particle opticalphoton
# /gps/energy 3.024 eV
# /gps/pos/type Plane
# /gps/pos/shape Circle
# /gps/pos/centre 0 0 19 cm
# /gps/pos/radius 254 mm
# /gps/ang/type iso
# /gps/ang/maxtheta   0 deg
# /gps/ang/mintheta   90 deg

## Gun mode ##
/gps/particle ion
/gps/ion {}
/gps/energy {} MeV

/gps/pos/type Plane
/gps/pos/shape Circle
/gps/pos/rot1 0 0 1
/gps/pos/rot2 1 0 0
/gps/pos/radius 2.5 cm
#/gps/pos/halfx 2.5 cm
#/gps/pos/halfy 2.5 cm
/gps/pos/centre 0 10 0 cm

/gps/direction 0 -1 0 mm

/run/beamOn {}
"""
    name_job_file = "./job.sh"
    name_sub_file = "./sub.sh"
    text_job = \
"""#!/bin/bash
#cd .. && source setup.sh && cd build &&
dir_work=/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/
seed=$(($1+1))
ion=$2
d_PMT=2
source $dir_work/setup.sh
( time $dir_work/build/Neutrino -mac """+template_outfile.format("${ion}")+""" -output root/${ion}_${d_PMT}cm_$seed.root -seed $seed -d_PMT ${d_PMT} -optical -UseTank -UseAcrylic -r_LS 2.5 -L_LS 2)
"""

    #### Check directories
    from Bash_Tool import CheckDirectory
    CheckDirectory(os.path.dirname(template_outfile))

    with open(name_sub_file, "w") as f_sub:
        for i in range(len(dir_ion["name"])):
            mac_file = template_outfile.format(dir_ion["name"][i])
            N = float(dir_ion["input_list"][i].split(" ")[1])

            # gen_scripts.GenMacFileCertainE(mac_file, dir_ion["input_list"][i], Ecertain=N*dir_ion["BeamE[MeV/u]"][i],nEvts=10000)

            with open(mac_file, "w") as f:
                f.write(template_mac_file.format(dir_ion["input_list"][i],N*dir_ion["BeamE[MeV/u]"][i], 5 ))

            with open(name_job_file, "w") as f:
                f.write(text_job)

            f_sub.write("hep_sub job.sh -argu \'%{ProcID}\' "+dir_ion["name"][i]+" -n 200   -o /dev/null -e /dev/null -m 4000 -wt short \n")
            # f_sub.write("hep_sub job.sh -argu \'%{ProcID}\' "+dir_ion["name"][i]+" -n 200   -o /dev/null -e /dev/null -m 4000 -wt short \n")
