# -*- coding:utf-8 -*-
# @Time: 2021/6/7 9:06
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenDetsimScripts.py
import matplotlib.pylab as plt
import numpy as np
import sys, os

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

class GenDetsimScripts:
    def __init__(self,short_jobs=False):
        self.template_detsim = ""
        self.debug_print = False
        self.i_added_sub_job=0
        self.template_sub_scripts = \
"""

job={}
chmod 755 $job
hep_sub $job -e /dev/null -o /dev/null """
        if short_jobs:
            self.template_sub_scripts += "-wt short"

    def SetTemplate(self, template):
        self.template_detsim:str = template

    def GenDetsimScripts(self,v_momentum:np.ndarray=np.zeros(3), v_position:np.ndarray=np.zeros(3), name_particle:str="neutron", name_label:str="",
                         name_dir_save="./"):
        if not os.path.exists(name_dir_save):
            os.makedirs(name_dir_save)
        p = np.sum(v_momentum**2)**0.5
        str_position = "--position "+" ".join([str(loc) for loc in v_position])
        str_momentum = "--momentums "+str(p)
        str_direction = "--directions "+ " ".join([str(momentum) for momentum in v_momentum])
        str_particles = f"--particles {name_particle}"
        self.options_add = f" {str_particles} {str_position} {str_direction} {str_momentum} "
        if self.debug_print:
            print(self.template_detsim.format(self.options_add))
        else:
            name_job = f"{name_dir_save}job_{name_label}.sh"
            if not os.path.isdir(name_dir_save):
                os.mkdir(name_dir_save)
            with open(name_job, "w") as f:
                f.write(self.template_detsim.format(self.options_add))
            self.AddJobToSubDetsimScript(name_dir_save=name_dir_save, name_job=name_job.split("/")[-1])

    def AddJobToSubDetsimScript(self, name_dir_save, name_job):
        if self.i_added_sub_job==0:
            option = "w"
        else:
            option = "a"
        with open(f"./sub.sh", option) as f:
            f.write(self.template_sub_scripts.format(f"{name_dir_save}{name_job}"))
        self.i_added_sub_job += 1


