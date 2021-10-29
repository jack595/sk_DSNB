# -*- coding:utf-8 -*-
# @Time: 2021/10/29 17:15
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: LoadDetsimEDM.py
# import matplotlib.pylab as plt
import numpy as np
# plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

import ROOT
ROOT.gSystem.Load("libEDMUtil")
ROOT.gSystem.Load("libSimEventV2")
# ROOT.gSystem.Load("libSimHeaderV2")
f = ROOT.TFile.Open("root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/PSD_LowE/alpha/detsim/root/detsim-992.root")
tree = f.Get("Event/Sim/SimEvent")
print(tree.GetEntries())
tree.GetEntry(1)
a = getattr(tree, "SimEvent", None)

v_cd_hits = a.getCDHitsVec()
for cd_hit in v_cd_hits[:10]:
    print("PMTID:\t",cd_hit.getPMTID()  )
    print("Hit-Time:\t",cd_hit.getHitTime())
    print("Npe:\t", cd_hit.getNPE())