# -*- coding:utf-8 -*-
# @Time: 2022/1/24 14:37
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: TimingConstantFit.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from LoadMultiFiles import LoadMultiROOTFiles

##################################### Load Data  ##########################################################################

# v_BeamEnergy = ["30MeV", "100MeV", "500MeV"]
v_BeamEnergy = ["500MeV"]

dir_PMT_diff_BeamE = {}
list_branch_filter = ["step_x", "step_y", "step_z"]
for BeamE in v_BeamEnergy:
    dir_PMT_diff_BeamE[BeamE] = LoadMultiROOTFiles(f"/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/proton/root/proton_{BeamE}_*.root",
                            name_branch="PMT_log", list_branch_filter=list_branch_filter)


######################################### Get emission time ##################################################
from HistTools import GetBinCenter
dir_v_time = {}

for beamE, dir_PMT in dir_PMT_diff_BeamE.items():
    v_emission_time = []
    v_n_hits = []
    for i_evt in range(len(dir_PMT["evtID"])):
        index_PMT_near = (dir_PMT["step_chamberID"][i_evt]==0)
        index_PMT_far = (dir_PMT["step_chamberID"][i_evt]==1)
        if np.any(index_PMT_far)==False:
            v_n_hits.append(0)
            continue
        h_time_near = np.histogram(dir_PMT["step_t"][i_evt][index_PMT_near], bins=np.linspace(0,20, 100))
        trigger_time = GetBinCenter(h_time_near[1])[h_time_near[0]>5][0]

        if len(dir_PMT["step_t"][i_evt][index_PMT_far])>0:
            v_emission_time.append(dir_PMT["step_t"][i_evt][index_PMT_far]-trigger_time)
            v_n_hits.append(len(dir_PMT["step_t"][i_evt][index_PMT_far]))
    v_emission_time = np.concatenate(v_emission_time)
    dir_v_time[beamE] = np.array(v_emission_time)


    plt.figure(1)
    h_time = plt.hist(v_emission_time,bins=np.linspace(0,250,250), histtype="step",
             density=True, label="Beam Energy = "+beamE+"\nEntries="+str(len(v_emission_time)))
    plt.xlabel("Time [ ns ]")
    plt.semilogy()
    plt.legend()

    plt.figure(2)
    plt.hist(v_n_hits, bins=range(0, 15), histtype='step',label="Beam Energy = "+beamE+
                                                                "\nEntries="+str(len(v_n_hits))+f", P={sum(v_n_hits)/len(v_n_hits):.2f}")
    plt.xlabel("N of Hits")
    plt.legend()

######################################### Fit time profile #################################################################
import ROOT
from RooFitTools import ArrayToTree

for beamE in dir_v_time.keys():
    xhigh = 250.
    tree_time_falling_edge = ArrayToTree(dir_v_time[beamE], "tt_0")

    ROOT.gSystem.Load("libRooFit")
    tt_0 = ROOT.RooRealVar("tt_0", "time [ns]", 40., xhigh)
    # ncharge = ROOT.RooRealVar("ncharge", "Q", 0., 100.)
    tau1 = ROOT.RooRealVar("tau1", "tau1", 20., 0., 800.)
    tau2 = ROOT.RooRealVar("tau2", "tau2", 150., 0., 800.)
    eta1 = ROOT.RooRealVar("eta1", "eta1", 0.7, 0., 1.)
    Nhit = ROOT.RooRealVar("Nhit", "Nhit", 10000, 1.e3, 1.e6)
    Ndark = ROOT.RooRealVar("Ndark", "Ndark", 0)

    lambda1 = ROOT.RooFormulaVar("lambda1", "lambda1", "-1./@0", ROOT.RooArgList(tau1))
    lambda2 = ROOT.RooFormulaVar("lambda2", "lambda2", "-1./@0", ROOT.RooArgList(tau2))
    #ROOT.RooFormulaVar lambda3("lambda3", "lambda3", "-1./@0", RooArgList(tau3))

    exp1 = ROOT.RooExponential("exp1", "exp1 distribution", tt_0, lambda1)
    exp2 = ROOT.RooExponential("exp2", "exp2 distribution", tt_0, lambda2)
    #ROOT.RooExponential exp3("exp3", "exp3 distribution", tt_0, lambda3)

    bpN1 = ROOT.RooFormulaVar("bpN1", "bpN1", '@0*@1', ROOT.RooArgList(Nhit, eta1))
    bpN2 = ROOT.RooFormulaVar("bpN2", "bpN2", '@0*(1.-@1)', ROOT.RooArgList(Nhit, eta1))
    #RooFormulaVar bpN3("bpN3", "bpN3", "@0*(1.-@1-@2)", RooArgList(Nhit, eta1, eta2))
    bpdark = ROOT.RooFormulaVar("bpdark", "bpdark", "@0*(900.-40.)", ROOT.RooArgList(Ndark))

    polybkg1 = ROOT.RooPolynomial("polybkg1", "bkg1 distribution", tt_0, ROOT.RooArgList())

    # ffpoly = ROOT.TF1("ffpoly", "[0]", -200, -100)
    # h_time_rising_edge.Fit(ffpoly, "RL")
    # double ndarkfit = ffpoly->GetParameter(0) * (xhigh - 40);
    # double ndarkerrfit = ffpoly->GetParError(0) * (xhigh - 40);
    # Ndark.setVal(ndarkfit);
    # Ndark.setError(ndarkerrfit);

    sum1 = ROOT.RooAddPdf("sum1", "sum1", ROOT.RooArgList(exp1, exp2, polybkg1), ROOT.RooArgList(bpN1, bpN2, Ndark))

    data = ROOT.RooDataSet("data", "data", ROOT.RooArgSet(tt_0), ROOT.RooFit.Import(tree_time_falling_edge))

    fitresult = sum1.fitTo(data, ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Extended(True))
    fitresult.Print()
