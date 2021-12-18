# -*- coding:utf-8 -*-
# @Time: 2021/12/16 16:19
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: FunctionFor_dE_dx.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
def GetDirForNoOpticalAnalyze(dir_events, dir_geninfo):
    v_dE_dx_average = []
    v_Equench = []
    v_Edep = []
    v_Einit = []
    v_dx = []
    pdgID_certain = dir_events["step_pdgID"][0][0]
    for i in range(len(dir_events["evtID"])):
        index_e = dir_events["step_pdgID"][i]==pdgID_certain
        dEquench = dir_events["step_Equench"][i][index_e]
        dEdep = dir_events["step_Edep"][i][index_e]
        if sum(dEquench)==0:
            print("Continue Edep=", np.sum(dEdep))
            continue
        dE_dx_average = sum(dEquench*dEdep/dir_events["step_dx"][i][index_e])/sum(dEquench)
        v_dE_dx_average.append(dE_dx_average)
        v_Equench.append(sum(dEquench))
        v_Edep.append(np.sum(dEdep))
        v_Einit.append(dir_geninfo["E_init"][dir_events["evtID"][i]])
        v_dx.append(sum(dir_events["step_dx"][i][index_e]))
    dir_return = {"Einit":np.array(v_Einit),
                  "dE_dx":np.array(v_dE_dx_average),
                  "Equench":np.array(v_Equench),
                  "Edep":np.array(v_Edep),
                  "dx":np.array(v_dx)}
    return pdgID_certain, dir_return
