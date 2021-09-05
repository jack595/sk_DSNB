# -*- coding:utf-8 -*-
# @Time: 2021/5/21 11:39
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotEventGif.py

import matplotlib.pylab as plt
import numpy as np
import ROOT
from tqdm import trange
from matplotlib.backends.backend_pdf import PdfPages
import uproot4 as up
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import pickle
import sys,os
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/")
from python_script.PlotTools import GetListOfCmap
from python_script.usgcnnTools import PMTIDMap, GetOneEventImage, CorrTOFByPMTID, GetRelativeDistribution, PlotRawSignal, PlotIntepSignal
from python_script.GetPhysicsProperty import PDGMassMap, GetKineticE
from python_script.GenDetsimScripts import GenDetsimScripts
from python_script.PlotTrackOfProcess import PlotTrackOfProcess
from copy import copy
from python_script.ClusteringTools import Clustering_SKM3D

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

        self.k_means_tool =  Clustering_SKM3D()
        self.do_clustering = False
        self.add_clustering_to_pdf = False
        self.clustering_with_weight = False

        self.debug_not_save_pdf = False
        self.save_into_one_pdf = True
        self.subtract_TOF = False
        self.constrain_by_subtract_TOF = False
        self.constrain_down_limit_by_subtract_TOF = -100
        self.constrain_up_limit_by_subtract_TOF = 2
        self.threshold_equen = 700
        self.study_atm = True
        self.study_atm_center_specific = False

        self.plot_result_interp = False

        self.plot_track = True

        self.only_large_pmt = True
        self.check_time_subtractTOF_distribution = False

        self.gen_detsim_scripts = GenDetsimScripts()
        self.gen_detsim = False
        self.plot_separate_particle = False
        self.relative_distribution_move_to_center = False
        self.use_interp_distribution = True
        self.plot_separate_particle_into_one_pdf = False # when this is false, outputs will be separate pdfs store different particle
        self.i_entry_to_separate = 0
        self.name_save_detsim = f"./SimultationStudy/entry_{self.i_entry_to_separate}/"

        self.load_detsim_userdata = True

        if self.plot_track:
            self.plot_track_tool = PlotTrackOfProcess()
            self.name_file_current_plot_track = ""
            self.name_file_template_plot_track = "/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/GenerateAtmNu_no_optical/detsim/detsim_user-{}.root"


    def LoadMeshFile(self, name_file_mesh:str):
        self.max_n_points_grid: bool = True
        self.do_calcgrid: bool = False
        p = pickle.load(open(name_file_mesh, "rb"))
        self.V = p['V']
        self.x_V, self.y_V, self.z_V = self.V[:, 0], self.V[:, 1], self.V[:, 2]
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
        self.name_files = name_files
        self.chain = ROOT.TChain(key_in_root)
        print(self.chain.Add(name_files))

    def ClearTChain(self):
        self.chain.Reset()

    def GenEquenList(self, name_save:str):
        self.chain.SetBranchStatus("*", 0)
        self.chain.SetBranchStatus("t_Qedep", 1)
        self.n_entries = self.chain.GetEntries()
        print("Total Entries:\t", self.n_entries)
        self.v_equen = []
        for i in trange(int(self.n_entries)):
            self.chain.GetEntry(i)
            self.v_equen.append(self.chain.t_Qedep)
        self.v_equen = np.array(self.v_equen)
        np.savez(f"./{name_save}", v_equen=self.v_equen)
        print(self.v_equen)
        self.chain.SetBranchStatus("*", 1)

    def LoadEquenList(self, name_save:str):
        f =np.load(name_save, allow_pickle=True)
        self.v_equen = f["v_equen"]

    def GetEntryListWithEquenCut(self):
        self.entries_list = np.where(self.v_equen>=self.threshold_equen)[0]

    def GetEntry(self, i:int):
        self.evt_pmt_level = {}
        self.evt_event_level = {}
        # self.entry_relative_one_file = self.chain.LoadTree(i)
        self.chain.GetEntry(i)

        if not self.load_detsim_userdata:
            self.seed_name_file_current_chain = str(self.chain.GetCurrentFile().GetName()).split("Uedm_")[1].split(".")[0]

        if self.load_detsim_userdata:
            self.evt_pmt_level["pmtid"] = np.array(self.chain.pmtID, dtype=np.int32)
            self.evt_pmt_level["npes"] = np.array(self.chain.nPE, dtype=np.float32)
            self.evt_pmt_level["hittime"] = np.array(self.chain.hitTime, dtype=np.float32)
            self.evt_event_level["equen"] = self.chain.edep
            self.evt_event_level["vertex"] = np.array([self.chain.edepX, self.chain.edepY, self.chain.edepZ])
            self.evt_event_level["evtID"] = self.chain.evtID
        else:
            self.evt_pmt_level["pmtid"] = np.array(self.chain.t_pmtid, dtype=np.int32)
            self.evt_pmt_level["npes"] = np.array(self.chain.t_npe, dtype=np.float32)
            self.evt_pmt_level["hittime"] = np.array(self.chain.t_hittime, dtype=np.float32)
            self.evt_event_level["equen"] = self.chain.t_Qedep
            self.evt_event_level["vertex"] = np.array([self.chain.t_QedepX, self.chain.t_QedepY, self.chain.t_QedepZ])
            if not self.study_atm_center_specific:
                self.evt_event_level["evtID"] = self.chain.t_evtID
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
        if self.constrain_by_subtract_TOF:
            self.evt_pmt_level["time_subtract_TOF"] = CorrTOFByPMTID(self.evt_pmt_level["hittime"], self.evt_pmt_level["pmtid"],
                                                           self.evt_event_level["vertex"], self.pmtmap)

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

    def CutByTimeSubtractTOF(self, time_subtract_TOF_down_limit:float, time_subtract_TOF_up_limit:float):
        check_time_distribution = False
        index_time_cut = (self.evt_pmt_level_after_cut["time_subtract_TOF"] < time_subtract_TOF_up_limit) & (self.evt_pmt_level_after_cut["time_subtract_TOF"]>=time_subtract_TOF_down_limit)
        for key in self.evt_pmt_level.keys():
            self.evt_pmt_level_after_cut[key] = self.evt_pmt_level_after_cut[key][index_time_cut]
        if check_time_distribution:
            plt.figure()
            plt.hist(self.evt_pmt_level["time_subtract_TOF"], bins=np.arange(-100, 500))
            plt.xlabel("$T_{hit}-T_{fly}$ [ ns ]")
            plt.show()

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

    def GetRawEventFig(self, event2dimage, x, y, z, plot_particles=False, relative_distribution=False,
                       name_title=""):
        if not relative_distribution:
            x_hittime, y_hittime, z_hittime = x,y,z
            x_equen, y_equen, z_equen = x, y, z
        else:
            x_hittime, y_hittime, z_hittime = x[0], y[0], z[0]
            x_equen, y_equen, z_equen = x[1], y[1], z[1]
        fig_hittime = plt.figure(self.time_label)
        ax1 = fig_hittime.add_subplot(121, projection='3d')
        indices = (event2dimage[1] != 0.)
        img_hittime = ax1.scatter(x_hittime[indices], y_hittime[indices], z_hittime[indices], c=event2dimage[1][indices], cmap=plt.hot(), s=1)
        if self.subtract_TOF:
            ax1.set_title("Emission Time")
        else:
            ax1.set_title("Hit-Time"+f"_{name_title}")
        fig_hittime.colorbar(img_hittime, orientation = 'horizontal')

        ax2 = fig_hittime.add_subplot(122, projection='3d')
        indices = (event2dimage[0] != 0)
        img_eqen = ax2.scatter(x_equen[indices], y_equen[indices], z_equen[indices], c=event2dimage[0][indices], cmap=plt.hot(), s=1)
        ax2.set_title("$E_{quen}$ "+self.time_label)
        # img_eqen = ax.scatter(x, y, z, c=event2dimage[0], cmap=plt.hot(), s=1)
        fig_hittime.colorbar(img_eqen, orientation = 'horizontal')
        self.PlotBaseSphere(ax1)
        self.PlotBaseSphere(ax2)
        if plot_particles:
            self.PlotParticles(fig_particles=fig_hittime, ax_particles=ax1,only_plot_vertex=True)
            self.PlotParticles(fig_particles=fig_hittime, ax_particles=ax2, only_plot_vertex=True)
        if self.debug_not_save_pdf:
            pass
            # plt.show()
            # plt.close()
        else:
            if self.save_into_one_pdf:
                self.pdf_total.savefig()
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
                if self.save_into_one_pdf:
                    self.pdf_total.savefig()
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
            p_max = np.max(v_p)
            # arrow = v2d_arrows[index_p_max]
            # ax_particles.arrow3D(arrow[0], arrow[1], arrow[2], arrow[3], arrow[4], arrow[5],
            #                      mutation_scale=20, arrowstyle="-|>",
            #                      linestyle='dashed')
            offset_text = 0
            step_offset = 30
            for i, arrow in enumerate(v2d_arrows):
                color = next(ax_particles._get_lines.prop_cycler)['color']
                ax_particles.arrow3D(arrow[0], arrow[1], arrow[2], arrow[3], arrow[4], arrow[5],
                                     mutation_scale=20,arrowstyle="-|>",
                                    linestyle='dashed', color=color)
                ax_particles.text(v2d_text_loc[i][0], v2d_text_loc[i][1], v2d_text_loc[i][2]+offset_text,str(int(self.evt_event_level["pdg"][i])), color=color )
                ax_particles.text(v2d_text_loc[i][0], v2d_text_loc[i][1], v2d_text_loc[i][2]-200+offset_text,"({:.0f}MeV)".format(self.evt_event_level["KineticE"][i]), color=color)
                vertex = self.evt_event_level["vertex"]
                offset_text += step_offset
            ax_particles.text(vertex[0], vertex[1],
                              vertex[2]+200*times_R,"("+" m, ".join("{:.1f}".format(item/1000) for item in self.evt_event_level["vertex"])+" m)",
                              color="blue")
            scale_axis_limit = 0.5
            ax_particles.set_xlim(vertex[0]-p_max*scale_axis_limit, vertex[0]+p_max*scale_axis_limit)
            ax_particles.set_ylim(vertex[1]-p_max*scale_axis_limit, vertex[1]+p_max*scale_axis_limit)
            ax_particles.set_zlim(vertex[2]-p_max*scale_axis_limit, vertex[2]+p_max*scale_axis_limit)
        ax_particles.scatter(self.evt_event_level["vertex"][0], self.evt_event_level["vertex"][1],
                             self.evt_event_level["vertex"][2], marker="*")

        if save_into_pdf:
            if self.save_into_one_pdf:
                self.pdf_total.savefig()
            self.pdf.savefig()
            plt.close()


    def  GetEventGif(self, i_entry=0, name_out_pdf:str="try.pdf", name_label:str="",
                     name_title=""):
        self.GetEntry(i_entry)
        # if self.evt_event_level["equen"]<self.threshold_equen:
        #     print(f"Equen lower than {self.threshold_equen}, Pass!!!")
        #     return
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
                time_bins = np.arange(0, 15, 1)
        else:
            # time_bins = np.arange(94, 1200, 1)
            if self.constrain_by_subtract_TOF:
                # time_bins = np.arange(0, 200, 15)
                time_bins = [0, 1250]
            else:
                time_bins = np.arange(0, 1250, 1)
        with PdfPages(name_out_pdf) as self.pdf:
            n_pdf = 0
            # if not load_detsim_userdata and not self.study_atm_center_specific:
            if self.plot_track:
                if not self.load_detsim_userdata:
                    self.PlotParticles(save_into_pdf=True)
                self.PlotTrack(name_title)
            for i in range(len(time_bins)-1):
                if n_pdf > 7:
                    break
                print("Processing ", time_bins[i], " ns")
                self.time_label = f"( {time_bins[i]} - {time_bins[i+1]} ns )"
                self.CutByHittime(hittime_cut_down_limit=time_bins[i], hittime_cut_up_limit=time_bins[i+1])
                if self.constrain_by_subtract_TOF:
                    self.CutByTimeSubtractTOF(time_subtract_TOF_down_limit=self.constrain_down_limit_by_subtract_TOF, time_subtract_TOF_up_limit=self.constrain_up_limit_by_subtract_TOF)
                if len(self.evt_pmt_level_after_cut["pmtid"])==0:
                    if self.plot_separate_particle_into_one_pdf:
                        self.set_diff_particle_eventimgs[name_label].append([])
                    continue
                (event2dimg, event2dimg_interp) = GetOneEventImage(self.evt_pmt_level_after_cut["pmtid"],
                                                                   self.evt_pmt_level_after_cut["hittime"],
                                                                   self.evt_pmt_level_after_cut["npes"],
                                                                   pmtmap=self.pmtmap, V=self.V,
                                                                   do_calcgrid=self.do_calcgrid,
                                                                   max_n_points_grid=self.max_n_points_grid,
                                                                   filter_near_zero=False)
                if self.plot_result_interp:
                    PlotRawSignal(event2dimg, self.x_raw_grid, self.y_raw_grid, self.z_raw_grid)
                    PlotIntepSignal(event2dimg_interp, self.x_V, self.y_V, self.z_V)
                    plt.show()
                    exit()

                if self.relative_distribution_move_to_center:
                    (eventimg_equen, x_relative_base_equen, y_relative_base_equen, z_relative_base_equen) =\
                    GetRelativeDistribution(event2dimg[0], self.x_raw_grid, self.y_raw_grid, self.z_raw_grid, self.evt_event_level["vertex"])
                    (eventimg_hittime, x_relative_base_hittime, y_relative_base_hittime, z_relative_base_hittime )= \
                    GetRelativeDistribution(event2dimg[1], self.x_raw_grid, self.y_raw_grid, self.z_raw_grid, self.evt_event_level["vertex"])
                    event2dimg_relative = np.array([eventimg_equen, eventimg_hittime])
                    x_relative, y_relative, z_relative = np.array([x_relative_base_equen, x_relative_base_hittime]), \
                                                        np.array([y_relative_base_equen, y_relative_base_hittime]),\
                                                        np.array([z_relative_base_equen, z_relative_base_hittime])

                if not self.plot_separate_particle_into_one_pdf or not self.plot_separate_particle:

                    def DoClustering(self):
                        if len(event2dimg[1][event2dimg[1]!=0])<10:
                            print("Stop to do clustering because too few points")
                            return
                        if self.relative_distribution_move_to_center:
                            self.k_means_tool.WorkFlowKMeans(event2dimg_relative, x_relative_base_equen,
                                                             y_relative_base_equen, z_relative_base_equen,
                                                             clustering_with_weight=self.clustering_with_weight)
                        else:
                            if self.use_interp_distribution:
                                self.k_means_tool.WorkFlowKMeans(event2dimg_interp, self.x_V*self.R, self.y_V*self.R,
                                                                 self.z_V*self.R,
                                                                 clustering_with_weight=self.clustering_with_weight)
                            else:
                                self.k_means_tool.WorkFlowKMeans(event2dimg, self.x_raw_grid, self.y_raw_grid,
                                                             self.z_raw_grid,
                                                             clustering_with_weight=self.clustering_with_weight)
                        if self.debug_not_save_pdf:
                            self.k_means_tool.PlotClusteredData()
                        else:
                            self.k_means_tool.PlotClusteredData(pdf_out=self.pdf_total)

                    if self.do_clustering:
                        DoClustering(self)
                        # self.k_means_tool.SetDatasetInterp(event2dimg_interp, self.x_V, self.y_V, self.z_V)
                       # print(self.k_means_tool.WorkFlowKMeans(event2dimg_interp, self.x_V, self.y_V, self.z_V))
                    else:
                        if self.relative_distribution_move_to_center:
                            self.GetRawEventFig(event2dimg_relative, x_relative, y_relative, z_relative,
                                                plot_particles=True, relative_distribution=self.relative_distribution_move_to_center)
                        else:
                            self.GetRawEventFig(event2dimg, self.x_raw_grid, self.y_raw_grid, self.z_raw_grid, plot_particles=True)
                        if self.add_clustering_to_pdf:
                            DoClustering(self)

                else:
                    self.set_diff_particle_eventimgs[name_label].append(copy(event2dimg))
                n_pdf += 1
    def CreatePDFTotal(self, name_pdf:str):
        self.pdf_total = PdfPages(name_pdf)
    def ClosePDFTotal(self):
        print("Closing PDF_total!")
        self.pdf_total.close()
        print("Closed PDF_total!")
        exit(1)

    def PlotTrack(self, name_title=""):
        if not self.load_detsim_userdata:
            if self.name_file_current_plot_track != self.name_file_template_plot_track.format(self.seed_name_file_current_chain):
                self.name_file_current_plot_track = self.name_file_template_plot_track.format(self.seed_name_file_current_chain)
                self.plot_track_tool.SetDataset(self.name_file_current_plot_track)
        else:
            self.plot_track_tool.SetDataset(self.name_files)
        self.plot_track_tool.PlotTrack(evtID_plot=self.evt_event_level["evtID"],pdf=self.pdf_total,
                                       threshold_track_length=100,debug=self.debug_not_save_pdf,show_p_direction=False,
                                       name_title=name_title)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot Event Gif')
    parser.add_argument("--file", "-f", type=str, help="file input to plot", default="root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/Atm/proton/0_0_12000/user-detsim-z_12000_theta_1.75.root")
    # parser.add_argument("--Nspe", "-N", type=int, default=0, help="this num is to control which ")
    args = parser.parse_args()

    use_equen_cut = False
    load_detsim_J20 = False
    # name_atm_file_study_center_specific = "/afs/ihep.ac.cn/users/h/huyuxiang/junofs/2021-2-23/proton-decay/test/atmsample/Uedm/Uedm_e_cc_coh.root"
    name_atm_file_study_center_specific = "/afs/ihep.ac.cn/users/h/huyuxiang/junofs/2021-2-23/proton-decay/test/atmsample/Uedm/Uedm_mu_cc_dis.root"
    if load_detsim_J20:
        gen_gif = GenEventGif("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v2r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv")
    else:
        gen_gif = GenEventGif(
        "/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre0/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv")
    gen_gif.LoadMeshFile(name_file_mesh="./mesh_files/icosphere_5.pkl")



    label_save_file = ""
    if gen_gif.subtract_TOF:
        label_save_file += "subtractTOF"
    if gen_gif.relative_distribution_move_to_center:
        label_save_file += "_relative"
    else:
        label_save_file += "_absolute"
    if use_equen_cut:
        label_save_file += f"_Eupper{gen_gif.threshold_equen}"


    if gen_gif.load_detsim_userdata:
        # gen_gif.LoadDataset("/afs/ihep.ac.cn/users/l/luoxj/scratchfs_juno_500G/e+_highE/user-detsim-50.root", "evt")
        # gen_gif.LoadDataset("/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/Sim_Single_Particle/e-/0_0_0/user-detsim-z_0_theta_0.00.root", "evt")
        # gen_gif.LoadDataset("root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/Atm/proton/0_0_12000/user-detsim-z_12000_theta_1.75.root", "evt")
        gen_gif.LoadDataset(args.file, "evt")
    else:
        if gen_gif.study_atm:
            if gen_gif.study_atm_center_specific:
                gen_gif.LoadDataset(name_atm_file_study_center_specific, "simevent")
            else:
                name_files = "/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/Atm_hu/data1/Uedm_*.root"
                name_tree = "simevent"
                gen_gif.LoadDataset(name_files, name_tree)
        else:
            gen_gif.LoadDataset("/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/ProtonDecay_hu/Uedm_*.root", "simevent")

    name_source_particle = ""
    if gen_gif.load_detsim_userdata:
        # name_source_particle = gen_gif.name_files.split("Sim_Single_Particle/")[-1].split("/")[0]
        name_source_particle = gen_gif.name_files.split("/eos/juno/user/luoxj/")[-1].split("/")[1] +gen_gif.name_files.split("user-detsim")[-1].split(".root")[0]
    else:
        if gen_gif.study_atm:
            if gen_gif.study_atm_center_specific:
                name_source_particle = name_atm_file_study_center_specific.split("/")[-1].split(".")[0]
            else:
                name_source_particle = "atm"
        else:
            name_source_particle = "proton-decay"

    if use_equen_cut:
        name_save_equen = "equen_list_"+name_source_particle+".npz"
        if not os.path.exists(name_save_equen):
            gen_gif.GenEquenList(name_save_equen)
        else:
            gen_gif.LoadEquenList(name_save_equen)
        gen_gif.GetEntryListWithEquenCut()
    else:
        gen_gif.entries_list = range(gen_gif.chain.GetEntries())


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
        if gen_gif.save_into_one_pdf:
            gen_gif.CreatePDFTotal(f"./plot_pdf/{name_source_particle}-{label_save_file}_total.pdf")
        # for i_entry in range(2, 40):
        # for i_entry in [0]:
        for i_entry in gen_gif.entries_list[:50]:
            gen_gif.GetEventGif(i_entry=i_entry, name_out_pdf=f"./plot_pdf/{name_source_particle}-{i_entry}_{label_save_file}.pdf")
        if gen_gif.save_into_one_pdf:
            gen_gif.ClosePDFTotal()
        # gen_gif.GetEventGif(i_entry=1, name_out_pdf=f"atm-NC-1.pdf")
