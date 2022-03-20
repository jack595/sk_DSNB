# -*- coding:utf-8 -*-
# @Time: 2021/12/7 18:48
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: CompareHittimeDistribution.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

import sys

import ROOT
if __name__ == "__main__":
    ROOT.gSystem.Load("libEDMUtil")
    ROOT.gSystem.Load("libSimEventV2")
    f = ROOT.TFile.Open("root://junoeos01.ihep.ac.cn//eos/juno/valprod/valprod4/PsdDataforDSNB_J21v1r0-Pre1/AtmNC/DS/det_sample_10467.root")
    tree = f.Get("Event/Sim/SimEvent")
    print(tree.GetEntries())
    tree.GetEntry(1)
    a = getattr(tree, "SimEvent", None)

    v_cd_hits = a.getCDHitsVec()
    for cd_hit in v_cd_hits:
        print("PMTID:\t",cd_hit.getPMTID()  )
        print("Hit-Time:\t",cd_hit.getHitTime())
        print("Npe:\t", cd_hit.getNPE())


