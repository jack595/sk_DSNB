# -*- coding:utf-8 -*-
# @Time: 2021/9/18 11:41
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GetUnrunningJobs.py
import matplotlib.pylab as plt
import numpy as np
import glob
import re
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import os
import subprocess

class GetUnrunningJobs:
    def __init__(self, dir_full_path:str):
        self.is_eos_case = ("/eos/" in dir_full_path)
        if "/eos/" in dir_full_path:
            self.files_list = subprocess.getstatusoutput(f"eos ls {dir_full_path}")[1].split("\n")
        else:
            self.files_list = glob.glob(dir_full_path)
        print("Files List[:10]:\t",self.files_list[:10])
        self.file_Num_exist = []
    def ExtractFileNumber(self, offset_file_num=0):
        for name_file_txt in self.files_list:
            if not self.is_eos_case:
                if "txt" in self.files_list[0]:
                    self.file_Num_exist.append(int(re.findall(r'\b\d+\b',name_file_txt.split("/")[-1].split(".txt")[0])[-1]))
                elif "root" in self.files_list[0]:
                    self.file_Num_exist.append(int(name_file_txt.split(".root")[0].split("_")[-1]))
                else:
                    print("Path about txt or root files should be input!!!!!!!!!!!")
                    exit(1)
            else:
                self.file_Num_exist.append( int(name_file_txt.split(".root")[0].split("_")[-1]))
        self.file_Num_exist = np.array(self.file_Num_exist) - offset_file_num
        return self.file_Num_exist
    def MakeUpRestFileNum(self, max_num_name_file:int):
        self.file_num_not_exist = list(set(range(max_num_name_file))-set(self.file_Num_exist))

    def GenJobsScripts(self,template_jobs:str,max_num_name_file:int, name_sub_jobs:str="sub_makeup.sh"):
        """
        From existing successful files to get complementary set, so find failed jobs to generate resubmit scripts
        :param template_jobs:
        :param max_num_name_file:
        :param name_sub_jobs:
        :return:
        """

        self.ExtractFileNumber()
        self.MakeUpRestFileNum(max_num_name_file)
        with open(name_sub_jobs, "w") as f:
            for i in self.file_num_not_exist:
                f.write(template_jobs.format(i))

    def GenJobsScriptsFromFailFiles(self,template_jobs:str, name_sub_jobs:str="sub_makeup.sh", offset_file_num=0):
        """
        From existing failed files to generate resubmit scripts
        :param template_jobs:
        :param name_sub_jobs:
        :return:
        """
        self.ExtractFileNumber(offset_file_num)
        print("Existing files number[:10]:\t",self.file_Num_exist[:10])
        with open(name_sub_jobs, "w") as f:
            for i in self.file_Num_exist:
                f.write(template_jobs.format(i))











