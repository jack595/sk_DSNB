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
def GetDirForNoOpticalAnalyze(dir_events, dir_geninfo, pdgID=None):
    v_dE_dx_average = []
    v_dE_dx_only_main_track = []
    v_Equench = []
    v_Edep = []
    v_Einit = []
    v_dx = []
    v_index = []
    if pdgID == None:
        from collections import Counter
        v_pdg_first = []
        for i in range(len(dir_events["step_pdgID"])):
            v_pdg_first.append(dir_events["step_pdgID"][i][0])
        pdgID_certain = Counter(v_pdg_first).most_common(1)[0][0]
    else:
        pdgID_certain = pdgID

    for i in range(len(dir_events["evtID"])):
        # index_e = dir_events["step_pdgID"][i]==pdgID_certain
        index_pdgID = dir_events["step_pdgID"][i]==pdgID_certain
        # dEquench = dir_events["step_Equench"][i][index_e]
        # dEdep = dir_events["step_Edep"][i][index_e]
        # dx =dir_events["step_dx"][i][index_e]

        dEquench = dir_events["step_Equench"][i]
        dEdep = dir_events["step_Edep"][i]
        dx =dir_events["step_dx"][i]

        if sum(dEquench)==0 or sum(dx[index_pdgID])==0:
            # print("Continue Edep=", np.sum(dEdep))
            continue
        dE_dx_average = sum(dEquench*dEdep/dx)/sum(dEquench)

        dE_dx_only_main_track = sum(dEdep)/sum(dx[index_pdgID])

        v_dE_dx_average.append(dE_dx_average)
        v_dE_dx_only_main_track.append(dE_dx_only_main_track)
        v_Equench.append(sum(dEquench))
        v_Edep.append(np.sum(dEdep))
        v_Einit.append(dir_geninfo["E_init"][dir_events["evtID"][i]])
        v_dx.append(sum(dir_events["step_dx"][i][index_pdgID]))
        v_index.append(i)

    dir_return = {"Einit":np.array(v_Einit),
                  "dE_dx":np.array(v_dE_dx_average),
                  "dE_dx_main_track":np.array(v_dE_dx_only_main_track),
                  "Equench":np.array(v_Equench),
                  "Edep":np.array(v_Edep),
                  "dx":np.array(v_dx),
                  "index":np.array(v_index)}
    return pdgID_certain, dir_return

def GetNPE(dir_PMT:dict, chamberID=0, mean=True):
    from collections import Counter
    v_nPE = []
    for i_evt in range(len(dir_PMT["evtID"])):
        index = (dir_PMT["step_pdgID"][i_evt]==20022) & (dir_PMT["step_chamberID"][i_evt]==chamberID)
        nPE = len( set(dir_PMT["step_trackID"][i_evt][index]) )
        v_nPE.append( nPE )
    if mean:
        return np.median(v_nPE)
    else:
        return np.array(v_nPE)

def GetSiNPE(dir_Si:dict, chamberID=0):
    from collections import Counter
    v_NPE = []
    for i_evt in range(len(dir_Si["evtID"])):
        index = (dir_Si["step_pdgID"][i_evt]==20022) & (dir_Si["step_chamberID"][i_evt]==chamberID)
        nPE = len( set(dir_Si["step_trackID"][i_evt][index]) )
        v_NPE.append( nPE )
    return np.array(v_NPE)

def GetSiPETime(dir_Si:dict, chamberID=0):
    v2d_PE_Time = []
    for i_evt in range(len(dir_Si["evtID"])):
        v_PE_Time = []
        index = (dir_Si["step_pdgID"][i_evt]==20022) & (dir_Si["step_chamberID"][i_evt]==chamberID)
        for trackID in set(dir_Si["step_trackID"][i_evt][index]):
            v_PE_Time.append( min( dir_Si["step_t"][i_evt][index][ dir_Si["step_trackID"][i_evt][index]==trackID ] ) )
        v2d_PE_Time.append( v_PE_Time )
    return v2d_PE_Time

