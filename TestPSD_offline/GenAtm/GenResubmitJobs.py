# -*- coding:utf-8 -*-
# @Time: 2021/12/9 15:15
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenResubmitJobs.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from GetUnrunningJobs import GetUnrunningJobs

if __name__ == "__main__":
    job_tool = GetUnrunningJobs(dir_full_path="./userdata_*.root")
    job_tool.GenJobsScriptsFromFailFiles(template_jobs="hep_sub jobs.sh -argu {}   -o /dev/null -e /dev/null -m 4000\n",
                            name_sub_jobs="./resub.sh",offset_file_num=0)
