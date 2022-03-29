# -*- coding:utf-8 -*-
# @Time: 2022/2/26 14:20
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotTimeProfile.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys
from matplotlib.backends.backend_pdf import PdfPages
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from LoadMultiFiles import LoadOneFileUproot
from PlotTools import LegendNotRepeated
from HistTools import GetBinCenter
import tqdm


class PlotTimeProfile:
    def __init__(self):
        self.dir_evts = {}
        self.dir_map = {}
        self.v_tags = ["pES", "eES", "AfterPulse"]
        self.v_colors = ["red", "blue", "green"]
        self.options_h_time = "_NotSubtractTOF"
        self.bins = np.loadtxt(f"/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/PSD//Bins_Setting{self.options_h_time}.txt",
                      delimiter=",")

        self.dir_pdf = {tag:PdfPages(f"{tag}{self.options_h_time}.pdf") for tag in self.v_tags}

    def LoadDataset(self, path_events:str="", path_map:str=""):
        self.dir_evts = LoadOneFileUproot(f"/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/PSD/root{self.options_h_time}/user_PSD_99__SN.root", name_branch="evt", return_list=False)
        self.dir_map = LoadOneFileUproot("/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/tag_event/root/sn_tag_99.root", name_branch='evtTruth', return_list=False)
        print("Events dict:\t", self.dir_evts.keys())
        print("Dict of map:\t", self.dir_map.keys())
    def PlotProfile(self):

        for time_type in ["h_time_without_charge"]:
            for i, tag in enumerate(self.v_tags):
                for j, v_time in tqdm.tqdm(enumerate(self.dir_evts["h_time_with_charge"][self.dir_map["evtType"]==tag])):
                    plt.figure()
                    h_time = v_time/np.diff(self.bins)
                    plt.plot(GetBinCenter(self.bins),h_time/np.max(h_time), linewidth=0.5, color=self.v_colors[i],label=tag)

                    # plt.semilogy()
                    LegendNotRepeated(bbox_to_anchor=(1,1))
                    # plt.legend(bbox_to_anchor=(1,1))
                    plt.xlabel("Time [ ns ]")
                    plt.title(time_type)
                    self.dir_pdf[tag].savefig()
                    plt.close()
                    if j>200:
                        break

    def ClosePDF(self):
        for key, pdf in self.dir_pdf.items():
            pdf.close()
if __name__ == '__main__':
    plot_tool = PlotTimeProfile()
    plot_tool.LoadDataset()
    plot_tool.PlotProfile()
    plot_tool.ClosePDF()
