# -*- coding:utf-8 -*-
# @Time: 2022/6/14 10:44
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenScriptsTo_pkl.py
import os

import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

if __name__ == "__main__":
    job_template = \
"""#!/bin/bash
particle=$1
fileNo_start=$2
fileNo_end=$3
directory=$4
source /afs/ihep.ac.cn/users/l/luoxj/junofs_500G/miniconda3/etc/profile.d/conda.sh && conda activate tf && 
python /afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/OneShotSimulation/AnalysisCode/TurnSimRootFileToDF.py \
-p $particle -s $fileNo_start -e $fileNo_end -d $directory \
-o /afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/OneShotSimulation/pkl/PMT_far_${particle}_${fileNo_start}.pkl

"""
    name_job = "job_to_pkl.sh"
    name_sub = "sub_to_pkl.sh"
    sub_template = \
"""hep_sub  {name_job} -argu {particle} {fileNo_start} {fileNo_end} {directory} -wt short -m 4000 -e /dev/null -o /dev/null\n"""

    v_name_to_gen = ["H_2","He_4",   "Li_6", "B_10", "C_12", "N_14","O_16", "F_18", "Ne_20", "Na_22"]
    v_calib = [1,0,1,0,1,0,1,0,1,0]

    with open(name_job,"w") as f:
        f.write(job_template)
    os.system(f"chmod 755 {name_job}")

    fileNo_start = 3001
    fileNo_end = 3600
    bins_fileNo = list(range(fileNo_start, fileNo_end,100))+[fileNo_end]
    with open(name_sub, "w") as f:
        for i, particle in enumerate( v_name_to_gen ):
            path_suffix = "_calib" if v_calib[i] else ''
            for i_start, i_end in zip(bins_fileNo[:-1], bins_fileNo[1:]):
                f.write(sub_template.format(name_job=name_job, particle=particle, fileNo_start=i_start,
                                            fileNo_end=i_end, directory="root2_addEvtIDinGen"+path_suffix))


