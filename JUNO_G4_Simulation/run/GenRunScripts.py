# -*- coding:utf-8 -*-
# @Time: 2021/12/12 10:11
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenRunScripts.py
import sys
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
import os

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

        self.template_mac_certain_energy =\
"""
/control/verbose 1
/run/verbose 1
/event/verbose 0
/tracking/verbose 0

## Gun mode ##
/gps/particle ion
/gps/ion {} 
/gps/energy {} MeV
/gps/position {} 10 0 cm
/gps/direction 0 -1 0 mm

/run/beamOn {}
"""
        self.template_job = \
f"""#!/bin/bash
dir_work={self.work_path}
"""+\
"""
source $dir_work/setup.sh 
$dir_work/build/Neutrino -mac {} -output {} -L_LS {} -L_tank {} {}

"""
        self.f_sub = open("sub.sh","w")
        
    def GenMacFile(self, outfile ,str_list_ion="", Emin:float=0, Emax:float=2000, nEvts:int=30000):
        with open(outfile, "w") as f:
            f.write(self.template_mac.format(str_list_ion,Emin, Emax, nEvts ))

    def GenMacFileCertainE(self, outfile, str_list_ion="", Ecertain:float=1000, nEvts:int=30000, pos_x=0):
        with open(outfile, "w") as f:
            f.write(self.template_mac_certain_energy.format(str_list_ion,Ecertain,pos_x, nEvts ))

    def GenScripts(self, input_mac_file ,out_root_file, name_scripts, L_LS=1, L_tank=1, options="", template_sub=None):
        with open(name_scripts, "w") as f:
            f.write(self.template_job.format(input_mac_file, out_root_file, L_LS, L_tank, options))
        # self.f_sub.write(f"hep_sub {name_scripts} -wt short -m 4000\n")
        if template_sub==None:
            self.f_sub.write(f"hep_sub {name_scripts} -m 4000\n")
        else:
            self.f_sub.write(template_sub.format(name_scripts))




if __name__=="__main__":
    import periodictable as pt

    list_elements_gen = list(pt.elements)[1:20]
    list_charge = range(1,len(list_elements_gen)+1)
    v_name_to_gen = []
    input_list = []

    DIY_particles_list = True
    if DIY_particles_list:
        from PeriodictableTools import PeriodictableTools
        table_tool = PeriodictableTools()
        v_name_to_gen = ["H_2",     "He_4",   "Li_6", "B_10", "C_12", "N_14","O_16", "F_18", "Ne_20", "Na_22"]
        for name in v_name_to_gen:
            Z = table_tool.MapToCharge(name.split("_")[0])
            N = name.split("_")[1]
            input_list.append(f"{Z} {N} {Z} 0")
    else:
        # for i_th,element in enumerate(list_elements_gen):
        #     for isotopes in element.isotopes:
        #         v_name_to_gen.append(f"{element.name}_{isotopes}")
        #         input_list.append(f"{i_th+1} {isotopes} {i_th+1} 0")
        for i_th,(element,charge) in enumerate(zip(list_elements_gen, list_charge)):
            j = 0
            for isotopes in element.isotopes:
                if j>1:
                    break
                if isotopes>= charge*2:
                    v_name_to_gen.append(f"{element.name}_{isotopes}")
                    input_list.append(f"{i_th+1} {isotopes} {i_th+1} 0")
                    j+=1



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
        Z = float(dir_ion["input_list"][i].split(" ")[1])
        gen_scripts.GenMacFile(mac_file, dir_ion["input_list"][i], Emin=Z*60,Emax=Z*350)
        gen_scripts.GenScripts(input_mac_file=mac_file, out_root_file=root_file, name_scripts=job_file)
        os.system("chmod 755 "+job_file)
