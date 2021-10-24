# -*- coding:utf-8 -*-
# @Time: 2021/7/1 9:47
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotTrackOfProcess.py
import matplotlib.pylab as plt
import numpy as np
import uproot4 as up
from matplotlib.backends.backend_pdf import PdfPages
import sys
from copy import copy
import tqdm
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
from GetPhysicsProperty import GetKineticE, PDGMassMap

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

class PlotTrackOfProcess:
    def __init__(self):
        self.set_evID = set()
        self.tree_track = None

        self.pdg_mass_map = PDGMassMap()
        self.pdg_mass_map.GetBaseMass()
        self.list_continue_pdg = [12, -12, 14, -14]
        self.name_file_last = ""

    def SetDataset(self, name_file:str, key_tree="mu_tracking"):
        if self.name_file_last != name_file:
            with up.open(name_file) as self.f:
                self.tree_track = self.f[key_tree]
                self.tree_evt = self.f["evt"]
                self.tree_depTree = self.f["depTree"]
                self.dir_tracks = {}
                for key in self.tree_track.keys():
                    self.dir_tracks[key] = np.array(self.tree_track[key])
                self.set_evID = set(self.dir_tracks["evtID"])
                self.v_equen = np.array(self.tree_depTree["QEnergyDeposit"])
                self.v_edep = np.array(self.tree_depTree["EnergyDeposit"])
                self.v_evtID_depTree = np.array(self.tree_depTree["evtID"])

                self.dir_evts = {}
                for key in self.tree_evt.keys():
                    self.dir_evts[key] = np.array(self.tree_evt[key])

                self.name_file_last = name_file
        else:
            pass


    def GetTotalEntries(self):
        return len(self.dir_evts["evtID"])

    def GetEquen(self, entry_source, filter_n_capture=False):
        if filter_n_capture:
            return np.sum(self.v_equen[entry_source][:-1])
        else:
            return np.sum(self.v_equen[entry_source])

    def GetEquenByEvtID(self, evtID, filter_n_capture=False):
        entry_source = np.where(self.v_evtID_depTree==evtID)[0]
        if filter_n_capture:
            return np.sum(self.v_equen[entry_source][:-1])
        else:
            return np.sum(self.v_equen[entry_source])

    def GetEdepFromEvt(self, entry_source):
        return self.dir_evts["edep"][entry_source]
    def GetEdep(self, entry_source, filter_n_capture=False):
        if filter_n_capture:
            return np.sum(self.v_edep[entry_source][:-1])
        else:
            return np.sum(self.v_edep[entry_source])

    def Get_v_Equen(self, filter_n_capture=False):
        self.v_equen_without_n_capture = []
        if filter_n_capture:
            for i in range(len(self.v_equen)):
                self.v_equen_without_n_capture.append(np.sum(self.v_equen[i][:-1]))
        else:
            for i in range(len(self.v_equen)):
                self.v_equen_without_n_capture.append(np.sum(self.v_equen[i]))
        return np.array(self.v_equen_without_n_capture)

    def GetEvtIDOfDepTree(self):
        return self.v_evtID_depTree

    def Get_v_Edep(self, filter_n_capture=False):
        self.v_edep_without_n_capture = []
        if filter_n_capture:
            for i in range(len(self.v_edep)):
                self.v_edep_without_n_capture.append(np.sum(self.v_edep[i][:-1]))
        else:
            for i in range(len(self.v_edep)):
                self.v_edep_without_n_capture.append(np.sum(self.v_edep[i]))
        return np.array(self.v_edep_without_n_capture)

    def PlotEquen(self, title="",bins=None, filter_n_capture=False, name_fig_save:str=None):
        check_filter_n_capture = False
        plt.figure()
        self.v_equen_without_n_capture = self.Get_v_Equen(filter_n_capture=filter_n_capture)
        if check_filter_n_capture and filter_n_capture:
            index_anormal_peak = (np.array(self.v_equen_without_n_capture)>5)
            number_anormal_peak = np.where(index_anormal_peak==True)[0]
            print("#######################################")
            # print("Equen:\t",self.v_equen[index_anormal_peak])
            # print("Edep:\t",self.dir_evts["edep"][index_anormal_peak])
            # print(self.Get_dE_dx_ByLoading_into_dir(number_anormal_peak[0], merge_same_pdg=True)[1])
            for i in range(5):
                print("dE:\t",self.Get_dE_Sum_into_dir(number_anormal_peak[i]))
                self.PlotTrackWithEntrySource(number_anormal_peak[i], debug=True, show_p_direction=False, print_track_info=True)

        plt.hist(self.v_equen_without_n_capture, bins=bins, histtype="step")
        plt.xlabel("$E_{quench}$ [ MeV ]")
        plt.title(title)
        if name_fig_save!=None:
            plt.savefig(name_fig_save)
        return np.array(self.v_equen_without_n_capture)

    def GetPDGSet(self,entry_source):
        evtID_plot = self.dir_evts["evtID"][entry_source]
        index_evtID_plot = (self.dir_tracks["evtID"] == evtID_plot)
        set_pdg = set()
        for i in range(len(self.dir_tracks["Mu_Posx"][index_evtID_plot])):
            pdg = self.dir_tracks["pdgID"][index_evtID_plot][i]
            set_pdg.add(pdg)
        return set_pdg

    def GetCertainPDGEkFromOneEvent(self, evtID:int, pdg:int, only_create_process=None):
        index_evtID = (self.dir_tracks["evtID"] == evtID)
        index_pdg_certain_evtID = ( (self.dir_tracks["pdgID"][index_evtID]==pdg) &
                                    (self.dir_tracks["MuCreateProcess"][index_evtID]==only_create_process) )
        mass = self.pdg_mass_map.PDGToMass(pdg)
        v_Ek_return = []
        for i in range(len(self.dir_tracks["Mu_Px"][index_evtID][index_pdg_certain_evtID])):
            one_track_px = self.dir_tracks["Mu_Px"][index_evtID][index_pdg_certain_evtID][i]
            one_track_py = self.dir_tracks["Mu_Py"][index_evtID][index_pdg_certain_evtID][i]
            one_track_pz = self.dir_tracks["Mu_Pz"][index_evtID][index_pdg_certain_evtID][i]
            p_square = (one_track_px[0] ** 2 + one_track_py[0] ** 2 + one_track_pz[0] ** 2)
            v_Ek_return.append(GetKineticE(p_square, mass))
        return np.array(v_Ek_return)

    def PlotTrack(self,evtID_plot, brief_show=True, pdf=None, debug=False, threshold_track_length=10, print_track_info=False,
                  show_p_direction=True, name_title="", ax=None,only_plot_parent_particle=False, show_process_name=False, plot_p_MeV=False):
        set_pdg = set()
        if brief_show:
            threshold_track_length_plot = threshold_track_length
        else:
            threshold_track_length_plot = 0
        if only_plot_parent_particle:
            threshold_track_length_plot = 0 # only parentID=0 particle, we don't need to omit any particles
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        ax.set_title(name_title)
        index_evtID_plot = (self.dir_tracks["evtID"] == evtID_plot)
        if print_track_info:
            print("####################################################")
        for i in range(len(self.dir_tracks["Mu_Posx"][index_evtID_plot])):
            if only_plot_parent_particle and self.dir_tracks["MuParentID"][index_evtID_plot][i]!=0 :
                continue
            pdg = self.dir_tracks["pdgID"][index_evtID_plot][i]
            if pdg in self.list_continue_pdg:
                continue
            color = next(ax._get_lines.prop_cycler)['color']
            one_track_x = self.dir_tracks["Mu_Posx"][index_evtID_plot][i]
            one_track_y = self.dir_tracks["Mu_Posy"][index_evtID_plot][i]
            one_track_z = self.dir_tracks["Mu_Posz"][index_evtID_plot][i]
            one_track_px = self.dir_tracks["Mu_Px"][index_evtID_plot][i]
            one_track_py = self.dir_tracks["Mu_Py"][index_evtID_plot][i]
            one_track_pz = self.dir_tracks["Mu_Pz"][index_evtID_plot][i]
            # color = cm.nipy_spectral( float(i) / len(one_track_x))
            l_track = ((one_track_x[0] - one_track_x[-1]) ** 2 +
                       (one_track_y[0] - one_track_y[-1]) ** 2 +
                       (one_track_z[0] - one_track_z[-1]) ** 2) ** 0.5
            if l_track > threshold_track_length_plot:
                p = (one_track_px[0] ** 2 + one_track_py[0] ** 2 + one_track_pz[0] ** 2) ** 0.5
                ax.plot(one_track_x, one_track_y, one_track_z, color=color)
                if show_p_direction:
                    ax.quiver(one_track_x, one_track_y, one_track_z,
                          one_track_px, one_track_py, one_track_pz,
                          color=color, ls="--", linewidth=2)
                # add_arrow(line, color=color)
                index_middle = int(len(one_track_x) / 2)
                if plot_p_MeV:
                    ax.text(one_track_x[index_middle], one_track_y[index_middle], one_track_z[index_middle],
                        f"{pdg}({1000*GetKineticE(p**2, self.pdg_mass_map.PDGToMass(pdg)):.1f}keV)", color=color)
                else:
                    ax.text(one_track_x[index_middle], one_track_y[index_middle], one_track_z[index_middle],
                            f"{pdg}", color=color)
                ax.set_xlabel("X [ mm ]")
                ax.set_ylabel("Y [ mm ]")
                ax.set_zlabel("Z [ mm ]")

                set_pdg.add(pdg)
                if print_track_info:
                    if show_process_name:
                        one_track_process_name = self.dir_tracks["MuCreateProcess"][index_evtID_plot][i]
                        print("pdg:\t", pdg, "\tmomentum:\t", p, "\tcreated process:\t", one_track_process_name,"\t, l_track:\t",l_track)
                    else:
                        print("pdg:\t", pdg, "\tmomentum:\t", p)
        if pdf != None and not debug:
            pdf.savefig()
        if not debug:
            plt.close()
        else:
            plt.show()
        return set_pdg

    def PlotTrackWithEntrySource(self, entry_source, brief_show=True, pdf=None, debug=False, threshold_track_length=10, print_track_info=False,show_p_direction=True,
                                 ax=None,only_plot_parent_particle=False, show_process_name=False,plot_p_MeV=False):
        evtID_to_plot = self.dir_evts["evtID"][entry_source]
        print("Edep:\t", self.dir_evts["edep"][entry_source], "---->")
        return self.PlotTrack(evtID_plot=evtID_to_plot, brief_show=brief_show, pdf=pdf, debug=debug, threshold_track_length=threshold_track_length, print_track_info=print_track_info,
                              show_p_direction=show_p_direction, ax=ax,only_plot_parent_particle=only_plot_parent_particle, show_process_name=show_process_name,plot_p_MeV=plot_p_MeV)

    def GetProcessNameWithEntrySource(self,entry_source):
        evtID_to_plot = self.dir_evts["evtID"][entry_source]
        index_evtID_plot = (self.dir_tracks["evtID"] == evtID_to_plot)
        one_track_process_name = self.dir_tracks["MuCreateProcess"][index_evtID_plot]
        return one_track_process_name

    def PlotIntoPdf(self, n_pages=10, name_out_pdf="track_plot.pdf"):
        with PdfPages(name_out_pdf) as pdf:
            for evtID_plot in self.set_evID[:n_pages]:
              self.PlotTrack(evtID_plot=evtID_plot, pdf=pdf)

    def GetMaxTrackLength(self, evtID_specific, vertex_Edep:np.ndarray):
        index_evtID_specific = (self.dir_tracks["evtID"] == evtID_specific)
        # print(self.dir_tracks["Mu_Posx"][index_evtID_specific])
        max_track_length_from_vertex = 0
        v_total_track_length = []
        for i in range(len(self.dir_tracks["Mu_Posx"][index_evtID_specific])):
            pdg = self.dir_tracks["pdgID"][index_evtID_specific][i]
            if pdg in self.list_continue_pdg:
                continue
            one_track_x = self.dir_tracks["Mu_Posx"][index_evtID_specific][i]
            one_track_y = self.dir_tracks["Mu_Posy"][index_evtID_specific][i]
            one_track_z = self.dir_tracks["Mu_Posz"][index_evtID_specific][i]
            one_track_px = self.dir_tracks["Mu_Px"][index_evtID_specific][i]
            one_track_py = self.dir_tracks["Mu_Py"][index_evtID_specific][i]
            one_track_pz = self.dir_tracks["Mu_Pz"][index_evtID_specific][i]

            # Skip Neutron Capture Gamma
            p = (one_track_px[0]**2+one_track_py[0]**2+one_track_pz[0]**2)**0.5
            if p<2.28 and p>2.18 and pdg==22:
                continue

            # mass = self.pdg_mass_map.PDGToMass(pdg)
            # Ek = GetKineticE(p_square, mass)
            # Skip Neutron Capture Gamma
            # if Ek < 2.3 and Ek > 2.15 and pdg==22:
            #     continue

            max_dl_one_track_from_vertex = np.max((one_track_x-vertex_Edep[0])**2+(one_track_y-vertex_Edep[1])**2+(one_track_z-vertex_Edep[2])**2)**0.5
            v_total_track_length.append( np.sum( (np.diff(one_track_x)**2+np.diff(one_track_y)**2+np.diff(one_track_z)**2)**0.5 ) )

            if max_dl_one_track_from_vertex > max_track_length_from_vertex:
                max_track_length_from_vertex = max_dl_one_track_from_vertex

        max_total_track_length = np.max(np.array(v_total_track_length))
        return (max_track_length_from_vertex, max_total_track_length)

    def GetMaxTrackLengthWithEntrySource(self,  entry_source, vertex_Edep:np.ndarray):
        evtID_to_plot = self.dir_evts["evtID"][entry_source]
        self.GetMaxTrackLength(evtID_to_plot, vertex_Edep)

    def GetOneTrack_dE_dx(self, one_track_x, one_track_y, one_track_z,
                          one_track_px, one_track_py, one_track_pz, mass):
        one_track_dx = []
        one_track_dE = []
        v_Ek =  GetKineticE(one_track_px**2+one_track_py**2+one_track_pz**2, mass)
        for i_step in range(len(one_track_x)-1):
            one_track_dx.append( ((one_track_x[i_step+1]-one_track_x[i_step])**2 + (one_track_y[i_step+1]-one_track_y[i_step])**2 +
            (one_track_z[i_step+1]-one_track_z[i_step])**2)**0.5 )
            # one_track_dE.append( ((one_track_px[i_step+1]-one_track_px[i_step])**2 + (one_track_py[i_step+1]-one_track_py[i_step])**2 +
            #                       (one_track_pz[i_step+1]-one_track_pz[i_step])**2)**0.5 )
            one_track_dE.append(v_Ek[i_step]-v_Ek[i_step+1])
        one_track_dE = np.array(one_track_dE)
        one_track_dx = np.array(one_track_dx)
        one_track_dE_dx = one_track_dE/one_track_dx
        # print("dE:\t", one_track_dE)
        # print("dx:\t",one_track_dx)
        return np.nan_to_num(one_track_dE_dx)

    def Get_dE_dx_ByCalculating(self, entry_source):
        index_evtID_specific = (self.dir_tracks["evtID"] == self.dir_evts["evtID"][entry_source])
        dir_dE_dx = {}
        for i in range(len(self.dir_tracks["Mu_Posx"][index_evtID_specific])):
            pdg = self.dir_tracks["pdgID"][index_evtID_specific][i]
            if pdg in self.list_continue_pdg:
                continue
            mass = self.pdg_mass_map.PDGToMass(pdg)
            one_track_x = self.dir_tracks["Mu_Posx"][index_evtID_specific][i]
            one_track_y = self.dir_tracks["Mu_Posy"][index_evtID_specific][i]
            one_track_z = self.dir_tracks["Mu_Posz"][index_evtID_specific][i]
            one_track_px = self.dir_tracks["Mu_Px"][index_evtID_specific][i]
            one_track_py = self.dir_tracks["Mu_Py"][index_evtID_specific][i]
            one_track_pz = self.dir_tracks["Mu_Pz"][index_evtID_specific][i]

            # Skip Neutron Capture Gamma
            p = (one_track_px[0]**2+one_track_py[0]**2+one_track_pz[0]**2)**0.5
            if p<2.28 and p>2.18 and pdg==22:
                continue

            # one_track_dE = (one_track_px[-1]**2+one_track_py[-1]**2+one_track_pz[-1]**2) - \
            #                (one_track_px[0]**2+one_track_py[0]**2+one_track_pz[0]**2)

            dir_dE_dx[f"{int(pdg)}_{i}"] = self.GetOneTrack_dE_dx(one_track_x,one_track_y, one_track_z,
                                         one_track_px, one_track_py, one_track_pz, mass)
        return dir_dE_dx


    def Get_dE_dx_ByLoading_into_dir(self, entry_source, merge_same_pdg=False, return_dE_ratio=False, return_dE_quench=False):
        index_evtID_specific = (self.dir_tracks["evtID"] == self.dir_evts["evtID"][entry_source])
        dir_dE_dx = {}
        dir_dE = {}
        dE_sum = 0
        dir_dE_quench = {}
        for i in range(len(self.dir_tracks["Mu_dE"][index_evtID_specific])):
            pdg = int(self.dir_tracks["pdgID"][index_evtID_specific][i])
            if pdg in self.list_continue_pdg:
                continue
            one_track_dE = self.dir_tracks["Mu_dE"][index_evtID_specific][i]
            one_track_dx = self.dir_tracks["Mu_dx"][index_evtID_specific][i]
            one_track_dE_quench = self.dir_tracks["Mu_dE_quench"][index_evtID_specific][i]
            one_track_dE_dx = np.nan_to_num(one_track_dE/one_track_dx)
            if not merge_same_pdg:
                dir_dE_dx[f"{pdg}_{i}"] = copy(one_track_dE_dx)
                dir_dE[f"{pdg}_{i}"] = copy(one_track_dE)
                dir_dE_quench[f"{pdg}_{i}"] = copy(one_track_dE_quench)
            else:
                if pdg not in dir_dE_dx:
                    dir_dE_dx[pdg] = np.array([])
                    dir_dE[pdg] = np.array([])
                    dir_dE_quench[pdg] = np.array([])
                dir_dE_dx[pdg] = np.concatenate((dir_dE_dx[pdg], one_track_dE_dx))
                dir_dE[pdg] = np.concatenate((dir_dE[pdg], one_track_dE))
                dir_dE_quench[pdg] = np.concatenate((dir_dE_quench[pdg],one_track_dE_quench))
            if return_dE_ratio:
                if return_dE_quench:
                    dE_sum += np.sum(one_track_dE_quench)
                else:
                    dE_sum += np.sum(one_track_dE)
        if return_dE_ratio:
            if return_dE_quench:
                for key in dir_dE_quench.keys():
                    dir_dE[key] = dir_dE_quench[key] /dE_sum
            else:
                for key in dir_dE.keys():
                    dir_dE[key] = dir_dE[key] / dE_sum
        # if return_dE_ratio==True, dir_dE is the ratio of dE contribution!!!!!!!!!
        if return_dE_quench:
            return (dir_dE_dx, dir_dE, dir_dE_quench)
        else:
            return (dir_dE_dx, dir_dE)

    def Get_dE_Sum_into_dir(self, entry_source):
        dir_v_dE = self.Get_dE_dx_ByLoading_into_dir(entry_source, merge_same_pdg=True)[1]
        dir_dE_sum = {}
        for key in dir_v_dE.keys():
            dir_dE_sum[key] = np.sum(dir_v_dE[key])
        return dir_dE_sum



    def PlotDiffParticle_dE_dx(self, entry_source, print_info=False, bins=None, filter_nuclei=True):
        dir_diff_particle_dE_dx, dir_diff_particle_dE = self.Get_dE_dx_ByLoading_into_dir(entry_source, merge_same_pdg=True)
        if print_info:
            print("dE/dx:\n",dir_diff_particle_dE_dx)
            print("dE:\n", dir_diff_particle_dE)
            total_dE_track = 0
            for key in dir_diff_particle_dE.keys():
                dE_particle = np.sum(dir_diff_particle_dE[key])
                print(key, dE_particle)
                total_dE_track += dE_particle
            print("total deposit energy with track:\t", total_dE_track)
            print("total deposit energy:\t", self.dir_evts["edep"][entry_source])
            print("total equen:\t", self.v_equen[entry_source])
            print("################################")

        plt.figure()
        for pdg in dir_diff_particle_dE_dx.keys():
            if filter_nuclei and pdg > 1e9:
                continue
            plt.hist(dir_diff_particle_dE_dx[pdg], bins=bins, histtype="step", label=pdg)
        plt.legend()

    def Get_dE_dx_ByLoading(self, entry_source,return_dE_quench=False):
        index_evtID_specific = (self.dir_tracks["evtID"] == self.dir_evts["evtID"][entry_source])
        v_dE_dx = []
        v_dE = []
        v_dE_quench = []
        for i in range(len(self.dir_tracks["Mu_dE"][index_evtID_specific])):
            one_track_dE = self.dir_tracks["Mu_dE"][index_evtID_specific][i]
            one_track_dx = self.dir_tracks["Mu_dx"][index_evtID_specific][i]
            one_track_dE_dx = np.nan_to_num(one_track_dE/one_track_dx)
            v_dE_dx.append(one_track_dE_dx)
            v_dE.append(one_track_dE)
            if return_dE_quench:
                one_track_dE_quench = self.dir_tracks["Mu_dE_quench"][index_evtID_specific][i]
                v_dE_quench.append(one_track_dE_quench)
        if return_dE_quench:
            return (np.concatenate(v_dE_dx), np.concatenate(v_dE), np.concatenate(v_dE_quench))
        else:
            return (np.concatenate(v_dE_dx), np.concatenate(v_dE))

    def Get_Average_dE_dx(self, entry_source, times_quench_factor=False):
        check_calculation = False
        if times_quench_factor:
            (v_dE_dx, v_dE, v_dE_quench) = self.Get_dE_dx_ByLoading(entry_source, return_dE_quench=True)
            sum_dE_quench = np.sum(v_dE_quench)
            dE_dx_average = np.sum(v_dE_dx*v_dE_quench)/sum_dE_quench
        else:
            (v_dE_dx, v_dE) = self.Get_dE_dx_ByLoading(entry_source)
            sum_dE = np.sum(v_dE)
            dE_dx_average = np.sum(v_dE_dx*v_dE)/sum_dE
        if check_calculation:
            print("dE/dx:\t",v_dE_dx)
            print("dE:\t",v_dE)
            print("dE/dx_average:\t",dE_dx_average)
        return dE_dx_average

    def Get_v_Average_dE_dx(self, entries_to_get=None, times_quench_factor=False):
        if entries_to_get == None:
            entries_to_get = self.GetTotalEntries()
        else:
            if entries_to_get>self.GetTotalEntries():
                entries_to_get = self.GetTotalEntries()
                print(f"WARNING:\tThis file got not enough entries for entries_to_get({entries_to_get})!!!!!")

        v_average_dE_dx = []
        for entry in tqdm.trange(entries_to_get):
            v_average_dE_dx.append(self.Get_Average_dE_dx(entry, times_quench_factor=times_quench_factor))
        return np.array(v_average_dE_dx)

    def Print_dE_dx_Contribution(self,entry_source):
        (dir_dE_dx, dir_dE_ratio, dir_dE_quench) = self.Get_dE_dx_ByLoading_into_dir(entry_source, merge_same_pdg=True, return_dE_ratio=True,
                                                                      return_dE_quench=True)
        print("\nAverage dE/dx:\t",self.Get_Average_dE_dx(entry_source, times_quench_factor=True))
        print("Equen:\t", self.GetEquen(entry_source,filter_n_capture=True),"\n")
        print("#######################################")
        for key in dir_dE_dx.keys():
            print(key, " --->")
            print("dE/dx:\t", dir_dE_dx[key])
            print("dE_ratio:", dir_dE_ratio[key])
            print("dE:\t", dir_dE_quench [key])
            print(f"Contributed dE/dx({key}):\t", np.sum(dir_dE_dx[key]*dir_dE_ratio[key]))
            print(f"Total Ratio({key})", np.sum(dir_dE_ratio[key]),"\n")
        print("#######################################")
















