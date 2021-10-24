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
    def __init__(self,short_jobs=False, seperate_job_into_subjobs=False):
        self.template_detsim = ""
        self.debug_print = False
        self.i_added_sub_job=0

        self.jobs_sub_option = ""
        self.use_specific_memory = True
        self.memory_to_specific = 4000
        self.seperate_job_into_subjobs = seperate_job_into_subjobs
        if self.use_specific_memory:
            self.jobs_sub_option += f"-m {self.memory_to_specific}"
        if seperate_job_into_subjobs:
            self.template_sub_scripts = \
"""

job={}
chmod 755 $job
hep_sub $job -argu {} \'%{{ProcID}}\' -n {} -e /dev/null -o /dev/null {}"""
        else:
            self.template_sub_scripts = \
"""

job={}
chmod 755 $job
hep_sub $job -e /dev/null -o /dev/null {}"""
        if short_jobs:
            self.template_sub_scripts += "-wt short"

    def SetTemplate(self, template):
        self.template_detsim:str = template

    def GenDetsimScriptsOfParticles(self,v_momentum:np.ndarray=np.zeros(3), v_position:np.ndarray=np.zeros(3), name_particle:str="neutron", name_label:str="",
                         name_dir_save="./", name_dir_save_data="./", n_events_in_one_job=0, n_subjobs=0):
        if "junoeos01.ihep.ac.cn" in name_dir_save_data:
            os.system("eos mkdir -p "+name_dir_save_data.split("root://junoeos01.ihep.ac.cn/")[-1])
        else:
            if not os.path.exists(name_dir_save_data):
                os.makedirs(name_dir_save_data)
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
                os.makedirs(name_dir_save)
            with open(name_job, "w") as f:
                f.write(self.template_detsim.format(self.options_add))
            if self.seperate_job_into_subjobs:
                self.AddJobToSubDetsimScript(name_dir_save=name_dir_save, name_job=name_job.split("/")[-1],n_events_in_one_job=n_events_in_one_job,
                                             n_subjobs=n_subjobs)
            else:
                self.AddJobToSubDetsimScript(name_dir_save=name_dir_save, name_job=name_job.split("/")[-1])
    def GenDetsimScriptsOfCalibrationSource(self, v_position: np.ndarray = np.zeros(3), name_particle: str = "AmC",
                                        name_label: str = "",
                                        name_dir_save="./", name_dir_save_data="./", add_source_shell=True,
                                            use_gendecay=False):
        if "junoeos01.ihep.ac.cn" in name_dir_save_data:
            os.system("eos mkdir -p "+name_dir_save_data.split("root://junoeos01.ihep.ac.cn/")[-1])
        else:
            if not os.path.exists(name_dir_save_data):
                os.makedirs(name_dir_save_data)
        str_position_calibration_shell = f" --source_weight_QC --OffsetInX {v_position[0]} --OffsetInY {v_position[1]} --OffsetInZ {v_position[2]}"
        str_position = "--global-position "+" ".join([str(loc) for loc in v_position])
        if not use_gendecay:
            str_particles = f"hepevt --exe {name_particle}"
        else:
            str_particles = f"gendecay --nuclear {name_particle}"
        if add_source_shell:
            self.options_add = f" {str_position_calibration_shell} "
        else:
            self.options_add = ""
        self.options_add += f" {str_particles} {str_position} "
        if self.debug_print:
            print(self.template_detsim.format(self.options_add))
        else:
            name_job = f"{name_dir_save}job_{name_label}.sh"
            if not os.path.isdir(name_dir_save):
                os.makedirs(name_dir_save)
            with open(name_job, "w") as f:
                f.write(self.template_detsim.format(self.options_add))
            self.AddJobToSubDetsimScript(name_dir_save=name_dir_save, name_job=name_job.split("/")[-1])


    def AddJobToSubDetsimScript(self, name_dir_save, name_job, n_events_in_one_job=0, n_subjobs=0):
        """

        :param name_dir_save:
        :param name_job:
        :param n_events_in_one_job: only used when self.seperate_job_into_subjobs
        :param n_subjobs: only used when self.seperate_job_into_subjobs
        :return:
        """
        if self.i_added_sub_job==0:
            option = "w"
        else:
            option = "a"
        with open(f"./sub.sh", option) as f:
            if self.seperate_job_into_subjobs:
                f.write(self.template_sub_scripts.format(f"{name_dir_save}{name_job}",n_events_in_one_job, n_subjobs,  self.jobs_sub_option))
            else:
                f.write(self.template_sub_scripts.format(f"{name_dir_save}{name_job}", self.jobs_sub_option))
        self.i_added_sub_job += 1


