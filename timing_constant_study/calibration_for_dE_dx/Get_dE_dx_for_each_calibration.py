# -*- coding:utf-8 -*-
# @Time: 2021/9/27 21:50
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: Get_dE_dx_for_each_calibration.py
import matplotlib.pylab as plt
import numpy as np

import sys
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

from PlotTrackOfProcess import PlotTrackOfProcess
track_tool = PlotTrackOfProcess()

import tqdm
import argparse
parser = argparse.ArgumentParser(description='Get dE/dx for calibration source')
parser.add_argument("--source", "-s", type=str, default="Co60",help="calibration source name")
par = parser.parse_args()

# v_particles = [ "Cs137", "Ge68", "C14", "K40", "Mn54", "Co60", "AmC"]
# v_name_files = ["user-detsim-_no_optical.root"]*len(v_particles)

particle = par.source
name_file = "user-detsim-_no_optical.root"
if particle == "Fe55":
    particle = "gamma_Fe55"
if particle=="AmC":
    neutron_filter = True
else:
    neutron_filter = False
template_path = \
    "root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/Calibration_for_time_constants/{}/no_optical/{}"
check_result_with_track = False

# template_path = \
# "/afs/ihep.ac.cn/users/l/luoxj/DSNB_component_fitting/timing_constant_study/{}/{}"
dir_event = {}
# for i,particle in enumerate(v_particles):

name_file_full_path = template_path.format(particle,
                        name_file)
# name_file_full_path = template_path.format(particle, v_name_files[i])
track_tool.SetDataset(name_file_full_path)
index_evtID_of_equen = track_tool.GetEvtIDOfDepTree()
entries = track_tool.GetTotalEntries()
v_dE_dx = []
v_dE_dx_with_quench = []
# v_equen = track_tool.PlotEquen(title=particle,
#                                name_fig_save=f"./figure/Equen_{particle}.png",
#                                filter_n_capture=v_neutron_filter[i], bins=200)
v_equen = track_tool.Get_v_Equen(filter_n_capture=neutron_filter)
# for j_entry in tqdm.trange(entries)[:1000]:
n_figure_track = 0
# entries = 100
index_evtID_of_equen = index_evtID_of_equen[(index_evtID_of_equen<entries)]
for j_entry in tqdm.trange(entries):

    if track_tool.GetEdepFromEvt(j_entry)==0:
        dE_dx_average = 0
        dE_dx_average_with_quench =  0
    else:
        dE_dx_average = track_tool.Get_Average_dE_dx(j_entry)
        dE_dx_average_with_quench = track_tool.Get_Average_dE_dx(j_entry,times_quench_factor=True)
    v_dE_dx.append(dE_dx_average)
    v_dE_dx_with_quench.append(dE_dx_average_with_quench)

    if check_result_with_track:

        track_tool.PlotTrackWithEntrySource(j_entry,print_track_info=True,
                                        debug=True, brief_show=False, show_p_direction=False)
        if n_figure_track>10:
            break
    n_figure_track += 1

dir_event["dE_dx_average"] = np.array(v_dE_dx)[index_evtID_of_equen]
dir_event["dE_dx_average_with_quench"] = np.array(v_dE_dx_with_quench)[index_evtID_of_equen]
dir_event["equen"] = np.array(v_equen)
np.savez(f"./{particle}/dE_dx_result.npz", dir_events=dir_event)
