# -*- coding:utf-8 -*-
# @Time: 2021/5/21 11:39
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotEventGif.py

import matplotlib.pylab as plt
import numpy as np
import ROOT
from matplotlib.backends.backend_pdf import PdfPages
# import uproot as up
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import pickle
import sys,os
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/")
from python_script.PlotTools import GetListOfCmap
from python_script.usgcnnTools import PMTIDMap, GetOneEventImage, PlotRawSignal, PlotIntepSignal, CorrTOF, CorrTOFByPMTID
from python_script.GetPhysicsProperty import PDGMassMap, GetKineticE
from python_script.GenDetsimScripts import GenDetsimScripts
from copy import copy

############### Set Arrow3D ########################################
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

########################################################################

class GenEventGif:
    def __init__(self, mapfile):
        self.evt_pmt_level = {}
        self.evt_pmt_level_after_cut = {}
        self.evt_event_level = {}
        self.set_diff_particle_eventimgs = {}
        self.pmtmap = PMTIDMap(mapfile)
        self.pdg_map = PDGMassMap()

        self.debug_not_save_pdf = False
        self.subtract_TOF = False
        self.study_atm = True

        self.only_large_pmt = True
        self.check_time_subtractTOF_distribution = False

        self.gen_detsim_scripts = GenDetsimScripts()
        self.gen_detsim = False
        self.plot_separate_particle = False
        self.plot_separate_particle_into_one_pdf = False # when this is false, outputs will be separate pdfs store different particle
        self.i_entry_to_separate = 0
        self.name_save_detsim = f"./SimultationStudy/entry_{self.i_entry_to_separate}/"



    def LoadMeshFile(self, name_file_mesh:str):
        self.max_n_points_grid: bool = True
        self.do_calcgrid: bool = False
        p = pickle.load(open(name_file_mesh, "rb"))
        self.V = p['V']
        n_grid = 128
        if self.max_n_points_grid:
            if self.do_calcgrid:
                self.pmtmap.CalcThetaPhiGrid()
            else:
                self.pmtmap.CalcThetaPhiPmtPoints()
        else:
            self.pmtmap.CalcThetaPhiSquareGrid(n_grid)
        if self.max_n_points_grid:
            if self.do_calcgrid:
                PHIS, THETAS = np.meshgrid(self.pmtmap.phis,
                                           self.pmtmap.thetas)  # Attention !!! Here we must be aware of the order of two inputs!!
            else:
                PHIS, THETAS = self.pmtmap.phis, self.pmtmap.thetas
        else:
            PHIS, THETAS = np.meshgrid(self.pmtmap.phis,
                                       self.pmtmap.thetas)  # Attention !!! Here we must be aware of the order of two inputs!!
            # print(f"thetas:{pmtmap.thetas}")
            # print(f"grid(thetas): {THETAS}")
        self.R = 17.7*1000 # m
        self.x_raw_grid = self.R * np.sin(THETAS) * np.cos(PHIS)
        self.y_raw_grid = self.R * np.sin(THETAS) * np.sin(PHIS)
        self.z_raw_grid = self.R * np.cos(THETAS)
        self.x_V, self.y_V, self.z_V = self.V[:, 0], self.V[:, 1], self.V[:, 2]

    def LoadDataset(self, name_files:str, key_in_root:str):
        self.chain = ROOT.TChain(key_in_root)
        self.chain.Add(name_files)

    def GetEntry(self, i:int):
        self.chain.GetEntry(i)
        if load_detsim_userdata:
            self.evt_pmt_level["pmtid"] = np.array(self.chain.pmtID, dtype=np.int32)
            self.evt_pmt_level["npes"] = np.array(self.chain.nPE, dtype=np.float32)
            self.evt_pmt_level["hittime"] = np.array(self.chain.hitTime, dtype=np.float32)
            self.evt_event_level["equen"] = self.chain.edep
            self.evt_event_level["vertex"] = np.array([self.chain.edepX, self.chain.edepY, self.chain.edepZ])
        else:
            self.evt_pmt_level["pmtid"] = np.array(self.chain.t_pmtid, dtype=np.int32)
            self.evt_pmt_level["npes"] = np.array(self.chain.t_npe, dtype=np.float32)
            self.evt_pmt_level["hittime"] = np.array(self.chain.t_hittime, dtype=np.float32)
            self.evt_event_level["equen"] = self.chain.t_Qedep
            self.evt_event_level["vertex"] = np.array([self.chain.t_QedepX, self.chain.t_QedepY, self.chain.t_QedepZ])
            self.evt_event_level["pdg"] = np.array(self.chain.t_pdgid)
            self.evt_event_level["init_px"] = np.array(self.chain.t_init_px)
            self.evt_event_level["init_py"] = np.array(self.chain.t_init_py)
            self.evt_event_level["init_pz"] = np.array(self.chain.t_init_pz)
            self.evt_event_level["p"] = np.sqrt(self.evt_event_level["init_px"] ** 2 + self.evt_event_level["init_py"] ** 2 + self.evt_event_level["init_pz"]**2)
            self.evt_event_level["mass"] = np.array([self.pdg_map.PDGToMass(pdg) for pdg in self.evt_event_level["pdg"]])
            self.evt_event_level["KineticE"] = np.array( GetKineticE(self.evt_event_level["p"]**2, self.evt_event_level["mass"]) )

        if self.only_large_pmt:
            self.index_large_pmt = self.evt_pmt_level["pmtid"]<self.pmtmap.maxpmtid
            for key in self.evt_pmt_level.keys():
                self.evt_pmt_level[key] = self.evt_pmt_level[key][self.index_large_pmt]
        if self.subtract_TOF:
            if self.check_time_subtractTOF_distribution:
                from copy import copy
                self.evt_pmt_level["hittime_raw"] = copy(self.evt_pmt_level["hittime"])
            self.evt_pmt_level["hittime"] = CorrTOFByPMTID(self.evt_pmt_level["hittime"], self.evt_pmt_level["pmtid"],
                                                           self.evt_event_level["vertex"], self.pmtmap)
            if self.check_time_subtractTOF_distribution:
                self.CheckHittimeDistribution()

    def InitializeEventImgSet(self):
        for label in self.v_name_label:
            self.set_diff_particle_eventimgs[label] = []

    def CheckHittimeDistribution(self):
        index_negative_time_subtractTOF = (self.evt_pmt_level["hittime"] < 0.)
        print("vertex:\t", self.evt_event_level["vertex"])
        print("Time:\t", self.evt_pmt_level["hittime"][index_negative_time_subtractTOF])
        v_pmt_pos = np.array([self.pmtmap.IdToPos(pmtid)[1:4] for pmtid in self.evt_pmt_level["pmtid"]])
        print("PMT loc:\t", v_pmt_pos[index_negative_time_subtractTOF])
        print("Hittime:\t", self.evt_pmt_level["hittime_raw"][index_negative_time_subtractTOF])

        plt.hist(self.evt_pmt_level["hittime"], bins=np.arange(-100, 1000, 1))
        plt.title("Emission Time")
        plt.xlabel("Time [ ns ]")
        plt.show()

    def CutByHittime(self, hittime_cut_down_limit:float, hittime_cut_up_limit:float):
        index_hittime_cut = (self.evt_pmt_level["hittime"]<hittime_cut_up_limit) & (self.evt_pmt_level["hittime"]>=hittime_cut_down_limit)
        for key in self.evt_pmt_level.keys():
            self.evt_pmt_level_after_cut[key] = self.evt_pmt_level[key][index_hittime_cut]

    def GenOneDetsimScript(self, name_label, v_momentum:np.ndarray=np.zeros(3), v_position:np.ndarray=np.zeros(3), name_particle:str="neutron", name_dir_save="./"):
        template = \
f'''#!/bin/bash
source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre0/setup.sh
(time python $TUTORIALROOT/share/tut_detsim.py --anamgr-normal-hit --evtmax 5 --seed 0 --output ./detsim-{name_label}.root --user-output  ./user-detsim-{name_label}.root --no-gdml gun '''\
+ "{}" + f") >& log-detsim-{name_label}.txt "
        self.gen_detsim_scripts.SetTemplate(template)
        self.gen_detsim_scripts.GenDetsimScripts(v_momentum, v_position, name_particle, name_label, name_dir_save)
    def SetDetsimLabel(self, entry_i):
        self.GetEntry(entry_i)
        pdg_nu = [12, 14, 16, 18]
        self.v_name_label = []
        self.v_vertex = []
        self.v_momentum = []
        self.v_name_particle = []
        for i,pdg in enumerate(self.evt_event_level["pdg"]):
            if pdg in pdg_nu:
                continue
            self.v_name_particle.append(self.pdg_map.PDGToName(pdg))
            self.v_momentum.append(np.array([self.evt_event_level["init_px"][i], self.evt_event_level["init_py"][i], self.evt_event_level["init_pz"][i]]))
            self.v_vertex.append(self.evt_event_level["vertex"])
            self.v_name_label.append(f"entry{entry_i}_pdgi{i}_{self.v_name_particle[-1]}_p_" +"_".join(["{:.1f}".format(p) for p in self.v_momentum[-1]]))

    def GenDetsimScriptsMainFunc(self):
        for i in range(len(self.v_name_label)):
            self.GenOneDetsimScript(self.v_name_label[i], self.v_momentum[i], self.v_vertex[i], self.v_name_particle[i], name_dir_save=self.name_save_detsim)



    def GetRawEventFig(self, event2dimage, x, y, z, plot_particles=False):
        fig_hittime = plt.figure(self.time_label)
        ax1 = fig_hittime.add_subplot(121, projection='3d')
        indices = (event2dimage[1] != 0.)
        img_hittime = ax1.scatter(x[indices], y[indices], z[indices], c=event2dimage[1][indices], cmap=plt.hot(), s=1)
        if self.subtract_TOF:
            ax1.set_title("Emission Time")
        else:
            ax1.set_title("Hit-Time")
        fig_hittime.colorbar(img_hittime, orientation = 'horizontal')

        ax2 = fig_hittime.add_subplot(122, projection='3d')
        indices = (event2dimage[0] != 0)
        img_eqen = ax2.scatter(x[indices], y[indices], z[indices], c=event2dimage[0][indices], cmap=plt.hot(), s=1)
        ax2.set_title("$E_{quen}$ "+self.time_label)
        # img_eqen = ax.scatter(x, y, z, c=event2dimage[0], cmap=plt.hot(), s=1)
        fig_hittime.colorbar(img_eqen, orientation = 'horizontal')
        self.PlotBaseSphere(ax1)
        self.PlotBaseSphere(ax2)
        if plot_particles:
            self.PlotParticles(fig_particles=fig_hittime, ax_particles=ax1,only_plot_vertex=True)
            self.PlotParticles(fig_particles=fig_hittime, ax_particles=ax2, only_plot_vertex=True)
        if self.debug_not_save_pdf:
            plt.show()
        else:
            self.pdf.savefig()
            plt.close()

    def GenRawDiffParticleEventsFigIntoOnePdf(self, x, y, z, plot_particles=False):
        v_cmap = GetListOfCmap()
        one_v_event2dimg = self.set_diff_particle_eventimgs[self.v_name_label[0]]
        for i in range(len(one_v_event2dimg)): # Time Loop
            fig_hittime = plt.figure(self.time_label)
            ax1 = fig_hittime.add_subplot(121, projection='3d')
            if self.subtract_TOF:
                ax1.set_title("Emission Time")
            else:
                ax1.set_title("Hit-Time")
            self.PlotBaseSphere(ax1)
            for (j, (key, v_event2dimg)) in enumerate(self.set_diff_particle_eventimgs.items()):
                event2dimage = v_event2dimg[i]
                if len(event2dimage) == 0:
                    continue
                indices_1 = (event2dimage[1] != 0)
                img_hittime = ax1.scatter(x[indices_1], y[indices_1], z[indices_1], c=event2dimage[1][indices_1],
                                          cmap=v_cmap[j], s=1)
                cbar_hittime = fig_hittime.colorbar(img_hittime, orientation = 'horizontal', pad=0.05, label=key)

            ax2 = fig_hittime.add_subplot(122, projection='3d')
            self.PlotBaseSphere(ax2)
            for (j, (key, v_event2dimg)) in enumerate(self.set_diff_particle_eventimgs.items()):
                event2dimage = v_event2dimg[i]
                indices_0 = (event2dimage[0] != 0)
                img_eqen = ax2.scatter(x[indices_0], y[indices_0], z[indices_0], c=event2dimage[0][indices_0],
                                       cmap=v_cmap[j], s=1)
                cbar_equen = fig_hittime.colorbar(img_eqen, orientation='horizontal',pad=0.05)

            if self.debug_not_save_pdf:
                plt.show()
            else:
                self.pdf.savefig()
                plt.close()

    def PlotBaseSphere(self, ax:Axes3D):
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)*self.R
        y = np.sin(u)*np.sin(v)*self.R
        z = np.cos(v)*self.R
        ax.plot_wireframe(x, y, z, color="black", linewidth=0.8, ls="--")

    def PlotParticles(self, fig_particles=None, ax_particles=None, save_into_pdf=False, times_R=1., only_plot_vertex=False):
        if fig_particles == None:
            fig_particles = plt.figure("particles")
        if ax_particles == None:
            ax_particles = fig_particles.add_subplot(111, projection='3d')

        if not only_plot_vertex:
            v2d_arrows = []
            v2d_text_loc = []
            v_p = []
            # print("vertex:\t", np.sum(self.evt_event_level["vertex"]**2)**0.5)
            # print(self.evt_event_level["vertex"])
            for i, pdg_label in enumerate(self.evt_event_level["pdg"]):
                v2d_arrows.append(np.concatenate((self.evt_event_level["vertex"],
                                                  times_R * np.array([self.evt_event_level["init_px"][i],
                                                                      self.evt_event_level["init_py"][i],
                                                                      self.evt_event_level["init_pz"][i]]))))
                v2d_text_loc.append(self.evt_event_level["vertex"] +
                                    times_R * np.array(
                    [self.evt_event_level["init_px"][i], self.evt_event_level["init_py"][i],
                     self.evt_event_level["init_pz"][i]]) / 3)

            for arrow in v2d_arrows:
                v_p.append(np.sum(arrow[-3:] ** 2) ** 0.5)

            index_p_max = np.argmax(v_p)
            arrow = v2d_arrows[index_p_max]
            ax_particles.arrow3D(arrow[0], arrow[1], arrow[2], arrow[3], arrow[4], arrow[5],
                                 mutation_scale=20, arrowstyle="-|>",
                                 linestyle='dashed')
            for i, arrow in enumerate(v2d_arrows):
                color = next(ax_particles._get_lines.prop_cycler)['color']
                ax_particles.arrow3D(arrow[0], arrow[1], arrow[2], arrow[3], arrow[4], arrow[5],
                                     mutation_scale=20,arrowstyle="-|>",
                                    linestyle='dashed', color=color)
                ax_particles.text(v2d_text_loc[i][0], v2d_text_loc[i][1], v2d_text_loc[i][2],str(int(self.evt_event_level["pdg"][i])), color=color )
                ax_particles.text(v2d_text_loc[i][0], v2d_text_loc[i][1], v2d_text_loc[i][2]-200,"({:.0f}MeV)".format(self.evt_event_level["KineticE"][i]), color=color)
                vertex = self.evt_event_level["vertex"]
            ax_particles.text(vertex[0], vertex[1],
                              vertex[2]+200*times_R,"("+" m, ".join("{:.1f}".format(item/1000) for item in self.evt_event_level["vertex"])+" m)",
                              color="blue")
        ax_particles.scatter(self.evt_event_level["vertex"][0], self.evt_event_level["vertex"][1],
                             self.evt_event_level["vertex"][2], marker="*")
        # self.PlotBaseSphere(ax_particles)
        # plt.show()
        if save_into_pdf:
            self.pdf.savefig()
            plt.close()


    def GetEventGif(self, i_entry=0, name_out_pdf:str="try.pdf", name_label:str=""):
        self.GetEntry(i_entry)
        if self.gen_detsim and not self.plot_separate_particle:
            self.SetDetsimLabel(i_entry)
            self.GenDetsimScriptsMainFunc()
        print("Printting ", name_out_pdf)
        print("############# Event Information #############################")
        print(self.evt_event_level)
        print("#############################################################")
        if self.subtract_TOF:
            if self.debug_not_save_pdf:
                time_bins = np.arange(5, 10, 1)
            else:
                time_bins = np.arange(0, 30, 1)
        else:
            time_bins = np.arange(0, 1200, 1)
        with PdfPages(name_out_pdf) as self.pdf:
            n_pdf = 0
            if not load_detsim_userdata:
                self.PlotParticles(save_into_pdf=True)
            for i in range(len(time_bins)-1):
                if n_pdf > 20:
                    break
                print("Processing ", time_bins[i], " ns")
                self.time_label = f"( {time_bins[i]} - {time_bins[i+1]} ns )"
                self.CutByHittime(hittime_cut_down_limit=time_bins[i], hittime_cut_up_limit=time_bins[i+1])
                if len(self.evt_pmt_level_after_cut["pmtid"])==0:
                    if self.plot_separate_particle_into_one_pdf:
                        self.set_diff_particle_eventimgs[name_label].append([])
                    continue
                (event2dimg, event2dimg_interp) = GetOneEventImage(self.evt_pmt_level_after_cut["pmtid"],
                                                                   self.evt_pmt_level_after_cut["hittime"],
                                                                   self.evt_pmt_level_after_cut["npes"],
                                                                   self.pmtmap, self.V,
                                                                   self.do_calcgrid, self.max_n_points_grid,
                                                                   subtract_TOF=self.subtract_TOF,
                                                                   event_vertex=self.evt_event_level["vertex"])
                if not self.plot_separate_particle_into_one_pdf or not self.plot_separate_particle:
                    if i == 0:
                        self.GetRawEventFig(event2dimg, self.x_raw_grid, self.y_raw_grid, self.z_raw_grid, plot_particles=True)
                    else:
                        self.GetRawEventFig(event2dimg, self.x_raw_grid, self.y_raw_grid, self.z_raw_grid, plot_particles=True)
                else:
                    self.set_diff_particle_eventimgs[name_label].append(copy(event2dimg))
                n_pdf += 1

if __name__ == '__main__':
    load_detsim_userdata = False
    if load_detsim_userdata:
        gen_gif = GenEventGif("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v2r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv")
    else:
        gen_gif = GenEventGif(
        "/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre0/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv")
    gen_gif.LoadMeshFile(name_file_mesh="./mesh_files/icosphere_5.pkl")

    name_source_particle = ""
    if load_detsim_userdata:
        name_source_particle = "e+"
    else:
        if gen_gif.study_atm:
            name_source_particle = "atm"
        else:
            name_source_particle = "proton-decay"
    label_save_file = ""
    if gen_gif.subtract_TOF:
        label_save_file += "subtractTOF"

    if load_detsim_userdata:
        gen_gif.LoadDataset("/afs/ihep.ac.cn/users/l/luoxj/scratchfs_juno_500G/e+_highE/user-detsim-50.root", "evt")
    else:
        if gen_gif.study_atm:
            gen_gif.LoadDataset("/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/Atm_hu/data1/Uedm_*.root", "simevent")
        else:
            gen_gif.LoadDataset("/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/ProtonDecay_hu/Uedm_*.root", "simevent")

    if gen_gif.plot_separate_particle:
        if not os.path.isdir(f"{gen_gif.name_save_detsim}/plot_pdf_separate/"):
            os.mkdir(f"{gen_gif.name_save_detsim}/plot_pdf_separate/")
        gen_gif.SetDetsimLabel(gen_gif.i_entry_to_separate)
        gen_gif.InitializeEventImgSet()
        load_detsim_userdata = True
        for i, name_label in enumerate(gen_gif.v_name_label):
            gen_gif.LoadDataset(f"{gen_gif.name_save_detsim}/user-detsim-{name_label}.root", "evt")
            gen_gif.GetEventGif(i_entry=0, name_out_pdf=f"{gen_gif.name_save_detsim}/plot_pdf_separate/{name_label}.pdf", name_label=name_label)
        if gen_gif.plot_separate_particle_into_one_pdf:
            with PdfPages(f"./plot_pdf/{name_source_particle}_{label_save_file}_separate_particles.pdf") as gen_gif.pdf:
                gen_gif.GenRawDiffParticleEventsFigIntoOnePdf(gen_gif.x_raw_grid, gen_gif.y_raw_grid, gen_gif.z_raw_grid)
    else:
        # for i_entry in [gen_gif.i_entry_to_separate]:
        for i_entry in range(0, 30):
            gen_gif.GetEventGif(i_entry=i_entry, name_out_pdf=f"./plot_pdf/{name_source_particle}-{i_entry}_{label_save_file}.pdf")
        # gen_gif.GetEventGif(i_entry=1, name_out_pdf=f"atm-NC-1.pdf")



