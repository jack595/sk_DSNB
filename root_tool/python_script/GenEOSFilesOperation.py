# -*- coding:utf-8 -*-
# @Time: 2021/11/30 16:06
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenEOSFilesOperation.py
import numpy as np
import sys
import os

def GenEOSRmFiles(path_file_list:str, output_file:str):
    v_files = np.loadtxt(path_file_list, dtype=str)
    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n")
        
        command_template = "eos newfind -f --name {} {}  | awk '{{print \"eos rm \" $1}}' | sh\n"
        for name_file in v_files:
            file_basename = os.path.basename(name_file.replace("*", ".*"))
            file_dirname = os.path.dirname(name_file)
            f.write(command_template.format(file_basename, file_dirname))
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Generate command to ls eos files")
    parser.add_argument("--file_list", "-f",  type=str, help="path of file_list.txt",required=True)
    parser.add_argument("--output_file", "-o",  type=str, help="path of output_file", default="ls_eos.sh")
    args = parser.parse_args()
    GenEOSRmFiles(args.file_list, args.output_file)
