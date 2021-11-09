# -*- coding:utf-8 -*-
# @Time: 2021/10/29 17:15
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: LoadDetsimEDM.py
# import matplotlib.pylab as plt
import os.path

import numpy as np
# plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

import ROOT

# v_path = np.loadtxt("./path_lib.txt",dtype=str)
# set_path = set()
# for path in v_path:
#     set_path.add(os.path.dirname(path))
# for path in set_path:
#     print(path)
#     ROOT.gSystem.AddDynamicPath(path)

# ROOT.gSystem.AddDynamicPath("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/offline/InstallArea/amd64_linux26/lib")
# ROOT.gSystem.AddDynamicPath("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/offline/InstallArea/python/")
# ROOT.gSystem.AddDynamicPath("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/mt.sniper/InstallArea/python/")
# ROOT.gSystem.AddDynamicPath("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/sniper/InstallArea/python/")
# ROOT.gSystem.AddDynamicPath("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/offline/InstallArea/amd64_linux26/lib/")
# ROOT.gSystem.AddDynamicPath("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/mt.sniper/InstallArea/amd64_linux26/lib/")
# ROOT.gSystem.AddDynamicPath("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/sniper/InstallArea/amd64_linux26/lib/")
# ROOT.gSystem.AddDynamicPath("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/ExternalLibs/ROOT/6.22.08/lib/")
# ROOT.gSystem.AddDynamicPath("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/ExternalLibs/CLHEP/2.4.1.0/lib/")
# ROOT.gSystem.AddDynamicPath("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/offline/DataModel/EvtNavigator/amd64_linux26/")
# print(ROOT.gSystem.ListLibraries())

ROOT.gSystem.Load("libEDMUtil")
ROOT.gSystem.Load("libSimEventV2")
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