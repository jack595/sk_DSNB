# -*- coding:utf-8 -*-
# @Time: 2021/9/18 14:52
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenResubmitJobs.py
import matplotlib.pylab as plt
import numpy as np
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/")
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

from python_script.GetUnrunningJobs import GetUnrunningJobs
if __name__ == "__main__":
    job_tool = GetUnrunningJobs(dir_full_path="/afs/ihep.ac.cn/users/l/luoxj/change_scintillation/GenAtmSimulation/DSNB/rec/log/*.txt\n")
    job_tool.GenJobsScripts(max_num_name_file=1000, template_jobs="hep_sub jobs.sh -argu {} 500 0  -o /dev/null -e /dev/null -m 4000",
                            name_sub_jobs="/afs/ihep.ac.cn/users/l/luoxj/DSNB_component_fitting/timing_constant_study/change_scintillation/GenAtmSimulation/DSNB/resub.sh")



import sys,os