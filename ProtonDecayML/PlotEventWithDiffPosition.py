# -*- coding:utf-8 -*-
# @Time: 2021/8/19 8:31
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotEventWithDiffPosition.py
import matplotlib.pylab as plt
import numpy as np
from PlotEventGif import GenEventGif

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
class PlotEventWithDiffPosition(GenEventGif):
    def __init__(self):
        super().__init__("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre0/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv")

    def PlotDiffPositionParticles(self,file_list_full_path:str):
        v_files = np.loadtxt(file_list_full_path, dtype=str)
        self.LoadMeshFile(name_file_mesh="./mesh_files/icosphere_5.pkl")
        self.CreatePDFTotal("./plot_pdf2/diff_particles_diff_position_total.pdf")
        print(v_files[:6])

        for file in v_files:
            try:
                self.LoadDataset(file, "evt")
                name_source_particle = file.split("/eos/juno/user/luoxj/Atm/")[-1].split("/")[0] +file.split("user-detsim")[-1].split(".root")[0]
                self.GetEventGif(i_entry=0, name_out_pdf=f"./plot_pdf2/{name_source_particle}.pdf", name_title=name_source_particle)
                # self.ClearTChain()
                print(file)
            except:
                continue
        self.ClosePDFTotal()


if __name__ == '__main__':
    plot_tool = PlotEventWithDiffPosition()
    plot_tool.PlotDiffPositionParticles("/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/Sim_Single_Particle/file_list.txt")
