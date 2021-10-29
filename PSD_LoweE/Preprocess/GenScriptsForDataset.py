# -*- coding:utf-8 -*-
# @Time: 2021/10/13 11:23
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenScriptsForDataset.py
import sys
import os
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
class GenScriptsForDataset:
    def __init__(self,path_scripts="./dataset_for_train/"):
        self.path_scripts = path_scripts
        # self.v_particles = ["alpha", "e-"]
        self.option = "_coti"
        self.v_particles = ["alpha", "e-"]
        self.v_n_jobs = [1600, 600]

        # mkdir
        if not os.path.isdir(self.path_scripts):
            for particle in self.v_particles:
                os.makedirs(self.path_scripts+f"{particle}/")

        self.template_jobs = \
"""
DISPLAY=""
source /afs/ihep.ac.cn/users/l/luoxj/junofs_500G/miniconda3/etc/profile.d/conda.sh && conda activate tf &&
python ../PSD_dataset.py -i "root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/PSD_LowE/$1/RawData{}/detsimNB_$2.root" -o "./$1$3/$2.npz" -t "$1"
"""
        self.template_sub = \
"""
chmod 755 job.sh
hep_sub job.sh -argu {} \'%{{ProcID}}\' {} -n {} -o /dev/null -e /dev/null -m 2500
"""
        self.template_test_job =\
"""
chmod 755 job.sh
./job.sh {} {} {}
"""
    def GenJobScripts(self):
        with open(f"{self.path_scripts}/job.sh", "w") as f:
            f.write(self.template_jobs.format(self.option))

        for i,particle in enumerate(self.v_particles):
            with open(f"{self.path_scripts}/sub_{particle}.sh", "w") as f:
                f.write(self.template_sub.format(particle, self.option,self.v_n_jobs[i]))
            with open(f"{self.path_scripts}/test_job_{particle}.sh", "w") as f:
                f.write(self.template_test_job.format(particle, 1, self.option))

if __name__ == '__main__':
    gen_scripts = GenScriptsForDataset()
    gen_scripts.GenJobScripts()



