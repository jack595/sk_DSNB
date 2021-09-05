# -*- coding:utf-8 -*-
# @Time: 2021/7/7 15:32
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenSimScripts.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys,os
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/")
from python_script.GenDetsimScripts import GenDetsimScripts
class GenSimScripts:
    """
    This script aims to generate different positions and different directions particles
    so that we can study direction's impact for detector response.
    """
    def __init__(self):
        self.v_vertex_z = np.arange(0, 17e3, 3000)
        self.v_direction_theta = np.linspace(0, np.pi, 10)
        self.gen_detsim_scripts = GenDetsimScripts(short_jobs=False)
        #self.v_name_partiles = ["mu-" , "e-"]
        #self.v_name_partiles = ["proton" , "neutron"]
        self.v_name_partiles = ["gamma","mu-" , "e-","proton" , "neutron","pi0", "pi+"]
        # self.v_name_partiles = []
        self.subdir_save_root = "Atm"
        self.use_split_function_detsim = False
        self.jobs_option_split = ""
        self.no_optical = True
        if self.no_optical:
            self.n_events_to_generate = 10000
        else:
            self.n_events_to_generate = 10
        if self.use_split_function_detsim:
            self.subdir_save_root += "_split"
            self.jobs_option_split = "--pmtsd-merge-twindow 1.0 --pmt-hit-type 2   --split-maxhits 100000"
        self.name_file_list = "file_list.txt"
        
    def GenOneDetsimScript(self, name_label, v_momentum:np.ndarray=np.zeros(3), v_position:np.ndarray=np.zeros(3), name_particle:str="neutron", name_dir_save="./",
                           name_dir_save_data="./"):
        if self.no_optical:
            self.jobs_option_split += " --no-optical"
        template = \
f'''#!/bin/bash
#source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre0/setup.sh
source /afs/ihep.ac.cn/users/l/luoxj/scratchfs_juno_500G/J21v1r0-Pre0/bashrc
(time python $TUTORIALROOT/share/tut_detsim.py --anamgr-normal-hit {self.jobs_option_split} --evtmax {self.n_events_to_generate} --seed 0 --output {name_dir_save_data}/detsim-{name_label}.root --user-output  {name_dir_save_data}/user-detsim-{name_label}.root --no-gdml gun '''\
+ "{}" + f") >& {name_dir_save}log-detsim-{name_label}.txt "
        self.gen_detsim_scripts.SetTemplate(template)
        self.gen_detsim_scripts.GenDetsimScripts(v_momentum, v_position, name_particle, name_label, name_dir_save,
                                                 name_dir_save_data)
    def GenDetsimScripts(self):
        self.InitialPlotEventCommandToScripts()
        for particle in self.v_name_partiles:
            for z in self.v_vertex_z:
                for theta in self.v_direction_theta:
                    name_label = f"z_{z:.0f}_theta_{theta:.2f}"
                    data_dir = f"root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/{self.subdir_save_root}/{particle}/0_0_{z:.0f}/"
                    if self.no_optical:
                        name_label += "_no_optical"
                        data_dir += "no_optical/"
                    self.GenOneDetsimScript(name_label, v_momentum=np.array([np.sin(theta), 0, np.cos(theta)])*1000,v_position=np.array([0,0,z]),
                                            name_particle=particle,name_dir_save=f"./{particle}/0_0_{z:.0f}/",
                                            name_dir_save_data= data_dir)
                    self.AddPlotEventCommandToScripts(f"{data_dir}/user-detsim-{name_label}.root")

    def InitialPlotEventCommandToScripts(self, name_job="job_plot_event.sh",name_job_sub="sub_plot_event.sh"):
        template_plot = \
'''#!/bin/bash
cd /afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/
source /afs/ihep.ac.cn/users/l/luoxj/junofs_500G/miniconda3/etc/profile.d/conda.sh && conda activate tf 
python PlotEventGif.py -f $1
'''
        with open(name_job, "w") as f:
            f.write(template_plot)
        with open(name_job_sub, "w") as f:
            f.write("DISPLAY=\"\"\n")
        with open(self.name_file_list, "w") as f:
            f.write("")

    def AddPlotEventCommandToScripts(self, file_full_path, name_job="job_plot_event.sh",name_job_sub="sub_plot_event.sh",
                                     ):
        template_plot_event = f"hep_sub {name_job} -argu {file_full_path} -o /dev/null -e /dev/null -m 4000\n"
        with open(self.name_file_list, "a") as f:
            f.write(f"{file_full_path}\n")
        with open(name_job_sub, "a") as f:
            f.write(template_plot_event)
if __name__ == '__main__':
    gen = GenSimScripts()
    gen.GenDetsimScripts()
