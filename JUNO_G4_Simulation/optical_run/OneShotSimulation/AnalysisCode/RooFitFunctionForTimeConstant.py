# -*- coding:utf-8 -*-
# @Time: 2022/6/11 9:58
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: RooFitFunctionForTimeConstant.py
import numpy as np

import sys

import tqdm

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

def FitTimeProfileWithTruth(name_pdf, dict_tree, label_constant):
    import ROOT
    t_upper = 1000
    dict_param = {"Ion":[], "tau1":[], "tau2":[], "tau3":[], "N1":[], "N2":[], "N3":[]}
    v_keys = dict_param.keys()
    for key in list(v_keys):
        if key!="Ion":
            dict_param[key+"_error"] = []

    t = ROOT.RooRealVar("time", "Time [ns]", 0, t_upper, "" )
    tau1 = ROOT.RooRealVar("#tau_{1}", "tau1", 4, 0., 20.)
    tau2 = ROOT.RooRealVar("#tau_{2}", "tau2", 25., 10., 100.)
    tau3 = ROOT.RooRealVar("#tau_{3}", "tau3", 200, 100., 800.)
    eta1 = ROOT.RooRealVar("#eta_{1}", "eta1", 0.7, 0., 1.)
    eta2 = ROOT.RooRealVar("#eta_{2}", "eta2", 0.2, 0., 1.)
    Nhit = ROOT.RooRealVar("N_{hit}", "Nhit", 1.e6, 1.e3, 5.e8)
    # Ndark = ROOT.RooRealVar("N_{dark}", "Ndark", 0)

    lambda1 = ROOT.RooFormulaVar ("lambda1", "lambda1", "-1./@0", ROOT.RooArgList(tau1))
    lambda2 = ROOT.RooFormulaVar ("lambda2", "lambda2", "-1./@0", ROOT.RooArgList(tau2))
    lambda3 = ROOT.RooFormulaVar ("lambda3", "lambda3", "-1./@0", ROOT.RooArgList(tau3))

    exp1 = ROOT.RooExponential ("exp1", "exp1 distribution", t, lambda1)
    exp2 = ROOT.RooExponential ("exp2", "exp2 distribution", t, lambda2)
    exp3 = ROOT.RooExponential ("exp3", "exp3 distribution", t, lambda3)

    bpN1 = ROOT.RooFormulaVar("bpN1", "bpN1", "@0*@1", ROOT.RooArgList(Nhit, eta1))
    bpN2 = ROOT.RooFormulaVar("bpN2", "bpN2", "@0*@1", ROOT.RooArgList(Nhit, eta2))
    bpN3 = ROOT.RooFormulaVar("bpN3", "bpN3", "@0*(1.-@1-@2)", ROOT.RooArgList(Nhit, eta1, eta2))
    # bpdark=ROOT.RooFormulaVar("bpdark", "bpdark", "@0*(900.-40.)", ROOT.RooArgList(Ndark))

    # polybkg1 = ROOT.RooPolynomial("polybkg1", "bkg1 distribution", t, ROOT.RooArgList());

    # ffpoly = TF1("ffpoly", "[0]", -200, -100)
    # h_time_rising_edge->Fit(ffpoly, "RL")
    # double ndarkfit = ffpoly->GetParameter(0) * (xhigh - 40);
    # double ndarkerrfit = ffpoly->GetParError(0) * (xhigh - 40);
    # Ndark.setVal(ndarkfit);
    # Ndark.setError(ndarkerrfit);

    sum1 = ROOT.RooAddPdf("sum1", "sum1", ROOT.RooArgList(exp1, exp2, exp3), ROOT.RooArgList(bpN1, bpN2, bpN3))

    c1 = ROOT.TCanvas("c_0")
    c1.Print(name_pdf+"[")
    for i,key in tqdm.tqdm( enumerate(dict_tree.keys()) ):
    # key = "H_2"
        data = ROOT.RooDataSet("data", "data", ROOT.RooArgSet(t), ROOT.RooFit.Import(dict_tree[key]))

        fitresult = sum1.fitTo(data, ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Extended(True))
        dict_param[label_constant].append(key)
        dict_param["N1"].append(eta1.getVal(0))
        dict_param["N2"].append(eta2.getVal(0))
        dict_param["N3"].append(1-(eta1.getVal(0)+eta2.getVal(0)))
        dict_param["tau1"].append(tau1.getVal(0))
        dict_param["tau2"].append(tau2.getVal(0))
        dict_param["tau3"].append(tau3.getVal(0))
        dict_param["N1_error"].append(eta1.getError())
        dict_param["N2_error"].append(eta2.getError())
        dict_param["N3_error"].append(pow(pow(eta1.getError(),2)+pow(eta2.getError(),2),0.5 ))
        dict_param["tau1_error"].append(tau1.getError())
        dict_param["tau2_error"].append(tau2.getError())
        dict_param["tau3_error"].append(tau3.getError())

        # fitresult.Print()

        xframe = t.frame(ROOT.RooFit.Title(key))
        #gPad->SetLogy();
        data.plotOn(xframe)
        # sum1.plotOn(xframe, ROOT.RooFit.Components(ROOT.RooArgSet(exp1)), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kBlue))
        # sum1.plotOn(xframe, ROOT.RooFit.Components(ROOT.RooArgSet(exp2)), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kBlue))
        # sum1.plotOn(xframe, ROOT.RooFit.Components(ROOT.RooArgSet(exp3)), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kBlue))
        # // sum1->plotOn(xframe, Components(RooArgSet(exp3)), LineStyle(kDashed), RooFit.LineColor(kOrange + 2));
        # sum1->plotOn(xframe, Components(RooArgSet(polybkg1)), LineStyle(kDotted), RooFit.LineColor(kGreen));
        sum1.plotOn(xframe, ROOT.RooFit.LineStyle(1), ROOT.RooFit.LineColor(2))
        sum1.paramOn(xframe)
        xframe.GetYaxis().SetRangeUser(0.1, 1000000)
        ROOT.gPad.SetLogy()

        xframe.Draw()
        c1.cd()
        c1.Print(name_pdf)
        # c1.SaveAs(f"/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/OneShotSimulation/AnalysisCode/fitResult/{key}.png")
        xframe.Delete()
        c1.Clear()

    c1.Print(name_pdf)
    c1.Print(name_pdf+"]")
    c1.Close()
    return dict_param