# -*- coding:utf-8 -*-
# @Time: 2021/12/12 10:11
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenRunScripts.py
import sys
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
import os
import periodictable as pt

class GenRunScripts:
    def __init__(self):
        self.work_path = "/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/"
        self.template_mac =\
"""
/control/verbose 1
/run/verbose 1
/event/verbose 0
/tracking/verbose 0

## Gun mode ##
/gps/particle ion
/gps/ion {} 
/gps/ene/type Lin
/gps/ene/min {} MeV
/gps/ene/max {} MeV
/gps/ene/gradient 0
/gps/ene/intercept 1
/gps/position 0 10 0 mm
/gps/direction 0 -1 0 mm

/run/beamOn {}
"""
        self.template_job = \
f"""#!/bin/bash
dir_work={self.work_path}
"""+\
"""
source $dir_work/setup.sh 
$dir_work/build/Neutrino -mac {} -output {}

"""
        self.f_sub = open("sub.sh","w")
        
    def GenMacFile(self, outfile ,str_list_ion="", Emin:float=100, Emax:float=2000, nEvts:int=20000):
        with open(outfile, "w") as f:
            f.write(self.template_mac.format(str_list_ion,Emin, Emax, nEvts ))

    def GenScripts(self, input_mac_file ,out_root_file, name_scripts ):
        with open(name_scripts, "w") as f:
            f.write(self.template_job.format(input_mac_file, out_root_file))
        self.f_sub.write(f"hep_sub {name_scripts} -wt short -m 2000\n")



if __name__=="__main__":
    list_elements_gen = list(pt.elements)[1:20]
    list_charge = range(1,len(list_elements_gen)+1)
    v_name_to_gen = []
    input_list = []
    for i_th,element in enumerate(list_elements_gen):
        for isotopes in element.isotopes:
            v_name_to_gen.append(f"{element.name}_{isotopes}")
            input_list.append(f"{i_th+1} {isotopes} {i_th+1} 0")

    dir_ion = \
    {
       "name":v_name_to_gen,
        "input_list":input_list
    }
    gen_scripts = GenRunScripts()
    template_outfile = gen_scripts.work_path+"/run/mac/{}.mac"
    template_outfile_root = gen_scripts.work_path+"/run/root/{}.root"
    template_outfile_job = gen_scripts.work_path+"/run/jobs/{}.sh"
    for i in range(len(dir_ion["name"])):
        mac_file = template_outfile.format(dir_ion["name"][i])
        root_file = template_outfile_root.format(dir_ion["name"][i])
        job_file = template_outfile_job.format(dir_ion["name"][i])
        gen_scripts.GenMacFile(mac_file, dir_ion["input_list"][i])
        gen_scripts.GenScripts(input_mac_file=mac_file, out_root_file=root_file, name_scripts=job_file)
        os.system("chmod 755 "+job_file)
