# -*- coding:utf-8 -*-
# @Time: 2022/7/6 10:19
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GetLeptonRation.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

import uproot as up
def GetLeptonRatio(path_file:str):
    with up.open(path_file) as f:
        v_Equench_capture = np.array( f["pdgdep"]["Qedep_Capture"] )
        v2d_PDGID = np.array(f["pdgdep"]["PDGID"])
        v2d_Qedep = np.array(f["pdgdep"]["Qedep"])

    for v_PDGID, v_Qdep, Equench_Capture in zip(v2d_PDGID, v2d_Qedep, v_Equench_capture):
        v_isLepton = np.array( [ True if PDGID in [11, -11, 22]  else False for PDGID in v_PDGID] )

        # Lepton
        Qedep_total_lepton = np.sum( v_Qdep[v_isLepton] )
        Qedep_prompt_lepton = Qedep_total_lepton - Equench_Capture

        # Hadron
        Qedep_total_hadron = np.sum(v_Qdep) - Qedep_total_lepton

        # Prompt Ratio
        lepton_ratio_prompt = np.nan_to_num( Qedep_prompt_lepton/(np.sum(v_Qdep)-Equench_Capture) )
        hadron_ratio_prompt = np.nan_to_num( Qedep_total_hadron/(np.sum(v_Qdep)-Equench_Capture) )
        print(lepton_ratio_prompt, hadron_ratio_prompt)



if __name__ == '__main__':
    path_file = "/afs/ihep.ac.cn/users/l/limin93/user_008400.root"
    GetLeptonRatio(path_file)
