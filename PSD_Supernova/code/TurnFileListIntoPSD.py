# -*- coding:utf-8 -*-
# @Time: 2022/2/23 20:16
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: TurnFileListIntoPSD.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys
import glob

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='Turn file list into PSD file list')
    # parser.add_argument("--input", "-i", type=str, default="filelist.txt", help="raw filelist")
    # parser.add_argument("--output", "-o", type=str, default="filelist_PSD.txt", help="path to save output filelist")
    #
    # arg = parser.parse_args()

    v_raw_filelist = glob.glob("/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/filelist/*.txt")
    path_save_output = "/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/filelist_PSD/"

    for filelist in v_raw_filelist:

        fileNo = filelist.split("fileList_")[1].split("th.txt")[0]

        v_filelist = np.loadtxt(filelist, delimiter="\n",dtype=str)

        v_PSD_filelist = []
        for i,file in enumerate(v_filelist):
            if (("cal" in file) or ("rec" in file) )and ("_user.root" not in file):
                v_PSD_filelist.append(file)
        with open(path_save_output+f"fileList_PSD_{fileNo}.txt","w") as f:
            for file in v_PSD_filelist:
                f.write(file+"\n")
    # np.savetxt(, v_PSD_filelist, delimiter="\n",dtype=str)

