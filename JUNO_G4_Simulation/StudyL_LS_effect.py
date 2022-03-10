# -*- coding:utf-8 -*-
# @Time: 2022/2/13 11:15
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: StudyL_LS_effect.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")


from LoadMultiFiles import LoadOneFileUproot
from copy import copy
filter_leaves = ['xyz_center', 'step_x', 'step_y', 'step_z',  'step_Edep', 'step_Equench', 'step_KineticE']

v_L_LS = ["1cm", "5cm", "5cm_no_tank"]
template_root_path = "/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/optical_run/alpha_{}/root/4000MeV_{}_120.root"
dir_LS = LoadOneFileUproot(template_root_path.format("1cm",34), name_branch="GdLS_log", return_list=False)
print("Total generated PE:\t", len(set(dir_LS["step_trackID"][0][dir_LS["step_pdgID"][0]==20022])))

# Study why collected PE decrease as L_LS increase
evtID = 0
index_opticalOP = dir_LS["step_pdgID"][evtID]==20022
v_trackID = dir_LS["step_trackID"][evtID][index_opticalOP]
set_trackID = set(v_trackID)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for trackID in list(set_trackID)[:100]:
    if len(dir_LS["step_x"][evtID][index_opticalOP][v_trackID==trackID])>5:
        n_step = 4
        ax.plot( dir_LS["step_x"][evtID][index_opticalOP][v_trackID==trackID][:n_step], dir_LS["step_y"][evtID][index_opticalOP][v_trackID==trackID][:n_step],
                 dir_LS["step_z"][evtID][index_opticalOP][v_trackID==trackID][:n_step],
                  linewidth=1)
        print(trackID)

    # else:
    #     plt.scatter( dir_LS["step_x"][evtID][index_opticalOP][v_trackID==trackID], dir_LS["step_y"][evtID][index_opticalOP][v_trackID==trackID],s=1 )
plt.show()