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
            self.f = up.open(name_file)
            self.tree_track = self.f[key_tree]
            self.tree_evt = self.f["evt"]
            self.dir_tracks = {}
            for key in self.tree_track.keys():
                self.dir_tracks[key] = np.array(self.tree_track[key])
            self.set_evID = set(self.dir_tracks["evtID"])

            self.dir_evts = {}
            for key in self.tree_evt.keys():
                self.dir_evts[key] = np.array(self.tree_evt[key])

            self.f.close()
            self.name_file_last = name_file
        else:
            pass


    def PlotTrack(self,evtID_plot, brief_show=True, pdf=None, debug=False, threshold_track_length=10, print_track_info=False):
        if brief_show:
            threshold_track_length_plot = threshold_track_length
        else:
            threshold_track_length_plot = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        index_evtID_plot = (self.dir_tracks["evtID"] == evtID_plot)
        # print(self.dir_tracks["Mu_Posx"][index_evtID_plot])
        if print_track_info:
            print("####################################################")
        for i in range(len(self.dir_tracks["Mu_Posx"][index_evtID_plot])):
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
                ax.quiver(one_track_x, one_track_y, one_track_z,
                          one_track_px, one_track_py, one_track_pz,
                          color=color, ls="--", linewidth=2)
                # add_arrow(line, color=color)
                index_middle = int(len(one_track_x) / 2)
                ax.text(one_track_x[index_middle], one_track_y[index_middle], one_track_z[index_middle],
                        pdg, color=color)
                # ax.text(one_track_x[index_middle], one_track_y[index_middle], one_track_z[index_middle]-200,"{:.1f}".format(p),color=color)
                ax.set_xlabel("X [ mm ]")
                ax.set_ylabel("Y [ mm ]")
                ax.set_zlabel("Z [ mm ]")

                if print_track_info:
                    print("pdg:\t", pdg, "\tmomentum:\t", p)
        if pdf != None and not debug:
            pdf.savefig()
        if not debug:
            plt.close()
        else:
            plt.show()

    def PlotTrackWithEntrySource(self, entry_source, brief_show=True, pdf=None, debug=False, threshold_track_length=10, print_track_info=False):
        evtID_to_plot = self.dir_evts["evtID"][entry_source]
        print("Edep:\t", self.dir_evts["edep"][entry_source], "---->")
        self.PlotTrack(evtID_plot=evtID_to_plot, brief_show=brief_show, pdf=pdf, debug=debug, threshold_track_length=threshold_track_length, print_track_info=print_track_info)

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







