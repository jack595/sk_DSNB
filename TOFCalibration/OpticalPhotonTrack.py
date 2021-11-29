# -*- coding:utf-8 -*-
# @Time: 2021/11/19 15:14
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: OpticalPhotonTrack.py
import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

from importlib import reload
import PlotTrackOfProcess
reload(PlotTrackOfProcess)
import uproot4 as up
from PlotTrackOfProcess import PlotTrackOfProcess
from PlotDetectorGeometry import PlotBaseSphere

import os
path_savefig = "/afs/ihep.ac.cn/users/l/luoxj/TOFCalibration/figure/"
if not os.path.exists(path_savefig):
    os.makedirs(path_savefig)

track_tool = PlotTrackOfProcess()
track_tool.SetPMTMap("/afs/ihep.ac.cn/users/l/luoxj/scratchfs_juno_500G/J21v2r0-trunk/data/Detector/Geometry/PMTPos_CD_LPMT.csv")

#%%

with np.load("/afs/ihep.ac.cn/users/l/luoxj/TOFCalibration/PMTToPlot.npz",allow_pickle=True) as f:
    dir_pmt = f["dir_pmt"].item()
list_pmtid = dir_pmt["PMTID"]


# list_z_plot = ["16526.6","0","-16526.6"]
# list_z_plot = ["0","-16526.6", "-17410"]
list_z_plot = ["-17410"]
for i,z in enumerate(list_z_plot):
    track_tool.SetDataset(f"root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/TOFCalibration/detsim/user-detsim-{z}.root")
    for j, pmtid in enumerate(list_pmtid):
        fig = plt.figure(j)
        ax = fig.add_subplot(111, projection='3d')

        PlotBaseSphere(ax,R=17.5e3)

        for i in range(3):
        # for i in [0]:
            track_tool.PlotOpticalTrack_HitCertainPMT(evtID=i, ax=ax,fig=fig,pmtID=pmtid,multiprocessing=False)

        # fig.savefig(path_savefig+f"track_of_optical_photon_{z}.png")
plt.show()
