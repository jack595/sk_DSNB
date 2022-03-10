#-*- coding:utf-8 -*-
# @Time: 2021/9/24 9:59
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: Bash_Tool.py
import os
import sys

import subprocess
def GetEOSFileList(dir_eos:str):
    return subprocess.getstatusoutput(f"eos ls {dir_eos}")[1].split("\n")

def CheckDirectory(path_dir:str):
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)


