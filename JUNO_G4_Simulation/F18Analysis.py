# -*- coding:utf-8 -*-
# @Time: 2022/1/27 15:41
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: F18Analysis.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")




from LoadMultiFiles import LoadOneFileUproot
path_file = "/afs/ihep.ac.cn/users/l/luoxj/JUNO_G4_Simulation/try_save_information.root"

dir_events = LoadOneFileUproot(path_file,
                               name_branch="GdLS_log",return_list=False)
dir_geninfo = LoadOneFileUproot(path_file,
                               name_branch="genInfo", return_list=False)
#%%
from collections import Counter
v_pdgID = []
for i in range(len(dir_events["step_pdgID"])):
    v_pdgID += list(set(dir_events["step_pdgID"][i]))
w = Counter(v_pdgID)
v_list = [str(key) for key in w.keys()]
plt.bar(v_list, w.values())
# plt.semilogy()
plt.title("Particles Species in Events")
plt.legend()
print(w)
print("Gamma Interaction Events Ratio:\t", w[22]/w[1000080180])
#%%
print(dir_events["step_dx"][0])
print(dir_events["step_pdgID"][0])
print(dir_events["step_Edep"][0])
from FunctionFor_dE_dx import GetDirForNoOpticalAnalyze
pdgIDD_certain, dir_analysis = GetDirForNoOpticalAnalyze(dir_events=dir_events, dir_geninfo=dir_geninfo)
print(dir_analysis.keys())

index_LeftBottom = dir_analysis["index"][(dir_analysis["Equench"]<0.05) & (dir_analysis["dE_dx"]<0.5)]
print(dir_events["step_pdgID"][index_LeftBottom][:5])
print(dir_events["step_dx"][index_LeftBottom][:5])
print(dir_events["step_trackID"][index_LeftBottom][:5])
print(dir_events.keys())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(index_LeftBottom)):
# for i in range(100):
    ax.plot(dir_events["step_x"][index_LeftBottom][i], dir_events["step_y"][index_LeftBottom][i],dir_events["step_z"][index_LeftBottom][i])
ax.set_xlabel("X [ mm ]")
ax.set_ylabel("Y [ mm ]")
ax.set_zlabel("Z [ mm ]")



plt.show()