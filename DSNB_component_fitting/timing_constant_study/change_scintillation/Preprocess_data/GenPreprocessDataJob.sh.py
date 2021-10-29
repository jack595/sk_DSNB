# -*- coding:utf-8 -*-
# @Time: 2021/9/9 0:00
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenPreprocessDataJob.sh.py
# import matplotlib.pylab as plt
import numpy as np
import os, sys, time


# plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
def GenRawDataJob(particle:str, type_time_constant:str, with_IBD:bool, work_dir:str):
    prefix_eos = "root://junoeos01.ihep.ac.cn/"
    input_dir = f"/eos/juno/user/luoxj/Sim_DSNB/{type_time_constant}/{particle}_0_0_0/"
    save_dir = input_dir+"RawData/"

    def MakeDir(dir):
      if not os.path.isdir(dir):
        try :
          os.makedirs(dir)
        except OSError:
          print('Failed to make dir :' + dir)
          sys.exit(-1)
    #workdir = '/junofs/users/chengj/workdir/PSD/Ana-new0817/atmNCana'
    #workdir = '/afs/ihep.ac.cn/users/l/luoxj/junofs_500G/DSNB_IBD_Selection//atmNCana'
    # workdir = os.getcwd()+f"/{type_time_constant}/{particle}/"
    jobdir = work_dir + '/job'
    outdirname = './'

    MakeDir(jobdir)
    os.chdir(jobdir)

    subname = jobdir + '/sub.sh'
    subfile = open(subname,'w')
    subfile.write('#!/bin/bash\n')
    # subfile.write('source /junofs/users/chengj/offEnvNew02sl6.sh \n')
    subfile.write('source /junofs/users/chengjie/offEnvNew02.sh \n')
    subfile.write('cd %s \n' %(jobdir))

    tag = 0
    for m in range(0, 1):
        model = int(m)
        #for i in range(801,901) :
        #for i in range(100,201) :
        for i in range(1,21) :
          jobname = jobdir + '/runatm_' + str(m) + '_' + str(i)  + '.sh'
          f = open(jobname, 'w')
          f.write('#!/bin/bash\n')
          # f.write('source /junofs/users/chengj/offEnvNew02sl6.sh \n')
          f.write('source /junofs/users/chengjie/offEnvNew02.sh \n')
          f.write('cd %s \n' %(workdir))
          for j in range(1,11) :
            tag = int(i-1)*10+j
            print(tag)
            seqno = str(tag).zfill(6)
            MakeDir(outdirname)
            outfilename = outdirname + '/ana_' +  seqno + '.root'
            # f.write('root -l -q anafs_nodark.C\"(%d, %d)\" \n' %(tag, model))
            if with_IBD:
                f.write(f'root -l -q anafs_data_with_IBD_Selection.C\'(%d, %d,\"{prefix_eos+input_dir}\", \"{save_dir}\")\' \n' %(tag, model))
            else:
                f.write(f'root -l -q anafs_data_without_IBD_Selection.C\'(%d, %d, \"{prefix_eos+input_dir}\", \"{save_dir}\")\' \n' %(tag, model))
            # f.write('root -l -q anafs.C\"(%d, %d)\" \n' %(tag, model))
            # f.write('root -l -q anafsHam.C\"(%d, %d)\" \n' %(tag, model))
            f.write('\n')
          f.close()
          subfile.write('chmod 755 runatm_%s_%s.sh \n' %(model,i))
          # subfile.write('hep_sub -os SL6 runatm_%s_%s.sh -g juno \n' %(model, i))
          #subfile.write('hep_sub -wt short -os CentOS7 runatm_%s_%s.sh -g juno \n' %(model, i))
          subfile.write('hep_sub -os CentOS7 runatm_%s_%s.sh -g juno \n' %(model, i))
    #  subfile.write('hep_sub rundsnbnew_%s.sh -g dyw \n' %(i))
          subfile.write('\n')

    subfile.close()
if __name__ == "__main__":
    v_particles = ["alpha", "gamma", "neutron"]
    v_types_time_constant = ["dE_dx_dependece", "three_component_J21"]
    dir_save = "/eos/juno/user/luoxj/Sim_DSNB/"
    dir_scripts = "/junofs/users/luoxj/DSNB_component_fitting/timing_constant_study/change_scintillation/Preprocess_data/"
    for type_time_constant in v_types_time_constant:
        for particle in v_particles:
            workdir = f"{dir_scripts}/{type_time_constant}/{particle}"
            if not os.path.exists(workdir):
                os.makedirs(workdir)
            if particle == "neutron":
                os.system(f"cp {dir_scripts}anafs_data_with_IBD_Selection.C {workdir}")
                # os.system(f"cp Job_with_IBD_Selection.py {workdir}")
                GenRawDataJob(particle, type_time_constant, with_IBD=True, work_dir=workdir)
            else:
                os.system(f"cp {dir_scripts}anafs_data_without_IBD_Selection.C {workdir}")
                # os.system(f"cp Job_without_IBD_Selection.py {workdir}")
                GenRawDataJob(particle, type_time_constant, with_IBD=False, work_dir=workdir)



            
        