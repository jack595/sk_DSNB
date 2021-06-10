# -*- coding:utf-8 -*-
# @Time: 2021/6/8 16:48
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GetPDGList_Atm.py
import matplotlib.pylab as plt
import numpy as np
import ROOT
import argparse

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

parser = argparse.ArgumentParser(description='Get PDG List')
parser.add_argument("--start", "-s", type=int, help="entry start", default=0 )
parser.add_argument("--end", "-e", type=int, help="entry end", default=1000)
parser.add_argument("--n_outfile", "-o", type=int, help="name of outfile", default=0)
args = parser.parse_args()

geninfo_chain = ROOT.TChain("PickEvt")
geninfo_chain.SetBranchStatus("*", 0)
geninfo_chain.SetBranchStatus("interType", 1)
geninfo_chain.SetBranchStatus("evtID", 1)
geninfo_chain.SetBranchStatus("ctag", 1)

dir_evts_save = {"filename":[], "entry":[],"evtID":[], "pdg":[]}

for i in range(args.start, args.end+1):
    geninfo_chain.Add(f"/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/Atm_hu/geninfo/geninfo_{i}.root")
entries_geninfo = geninfo_chain.GetEntries()

print("Total Entries:\t",entries_geninfo)
for i in range(entries_geninfo):
    geninfo_chain.GetEntry(i)
    # print(geninfo_chain.evtID,geninfo_chain.interType, geninfo_chain.ctag)
    if geninfo_chain.interType == 1 and geninfo_chain.ctag==2:
        # print(geninfo_chain.evtID,geninfo_chain.interType, geninfo_chain.ctag)
        # print(geninfo_chain.GetCurrentFile().GetName())
        # print(geninfo_chain.LoadTree(i))
        dir_evts_save["filename"].append(geninfo_chain.GetCurrentFile().GetName())
        dir_evts_save["pdg"].append(np.array(geninfo_chain.pdg))
        dir_evts_save["entry"].append(geninfo_chain.LoadTree(i))
        dir_evts_save["evtID"].append(geninfo_chain.evtID)
print(dir_evts_save)


