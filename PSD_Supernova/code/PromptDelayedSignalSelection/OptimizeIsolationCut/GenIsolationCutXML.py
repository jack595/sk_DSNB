# -*- coding:utf-8 -*-
# @Time: 2022/4/21 15:35
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GenIsolationCutXML.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from Bash_Tool import CheckDirectory

template_xml = \
"""<?xml version="1.0" encoding="UTF-8" ?>

<TAGLIST>
    <tag tagname="snIBD">
        <condition>
            <name>distance</name>
            <min>0</min>
            <max>{dR_cut}</max>
        </condition>
        <condition>
            <name>deltaT</name>
            <min>-{dT_cut}</min>
            <max>{dT_cut}</max>
        </condition>
        <condition>
            <name>VertexR</name>
            <max>17000</max>
        </condition>
    </tag>
</TAGLIST>"""

# v_dR_cut = np.arange(1, 10, 1) # m
# v_dt_cut = np.arange( 20, 40, 4 ) # ms

import argparse
parser = argparse.ArgumentParser(description="Generate configuration files for Isolation Selection")
parser.add_argument("--IsolationBeforeIBD",'-b', action="store_true", default=False,
                    help="Do Isolation selection before IBD Selection")
arg = parser.parse_args()

v_dR_cut = np.arange(1, 10, 1) # m
v_dt_cut = np.concatenate( (np.arange( 0.2, 1, 0.4 ),np.arange( 1, 4, 1 ),np.arange( 4, 20, 4 ) )) # ms

template_job_sub = \
"""hep_sub Job_IsotionSelection.sh -argu 0 {name_job_file} {IsolationBeforeIBD} -wt short -o {name_log_file} -e /dev/null\n"""

if arg.IsolationBeforeIBD:
    suffix = "_beforeIBD"
else:
    suffix = ""
CheckDirectory(f"./xml{suffix}/")
CheckDirectory(f"./log{suffix}/")


with open(f"sub{suffix}.sh","w") as f_sub:
    for dR_cut in v_dR_cut:
        for dt_cut in v_dt_cut:
            name_job_file = f"./xml{suffix}/IsolationCriteria_{dR_cut}m_{dt_cut}ms.xml"
            name_log_file = f"./log{suffix}/log_{dR_cut}m_{dt_cut}ms.txt"
            
            with open(name_job_file, "w") as f:
                f.write( template_xml.format(dT_cut=dt_cut*1e6, dR_cut=dR_cut*1000) )
            
            f_sub.write(template_job_sub.format(name_job_file=name_job_file, name_log_file=name_log_file,
                                                IsolationBeforeIBD=int(arg.IsolationBeforeIBD)) )



