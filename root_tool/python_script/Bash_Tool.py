#-*- coding:utf-8 -*-
# @Time: 2021/9/24 9:59
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: Bash_Tool.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

import subprocess
def GetEOSFileList(dir_eos:str):
    return subprocess.getstatusoutput(f"eos ls {dir_eos}")[1].split("\n")

