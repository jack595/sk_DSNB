# -*- coding:utf-8 -*-
# @Time: 2021/5/21 9:56
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: usgcnnTools.py
import matplotlib.pylab as plt
import numpy as np
import ROOT
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import pickle
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

class PMTIDMap():
    # The position of each pmt is stored in a root file different from the data file.
    NumPMT = 0
    maxpmtid = 17612
    thetaphi_dict = {}
    thetas = []

    # read the PMT map from the root file.
    def __init__(self, csvmap):
        pmtcsv = open(csvmap, 'r')
        self.pmtmap = {}
        for line in pmtcsv:
            pmt_instance = (line.split())
            self.pmtmap[int(pmt_instance[0])] = (
                int(pmt_instance[0]), float(pmt_instance[1]), float(pmt_instance[2]), float(pmt_instance[3]),
                float(pmt_instance[4]), float(pmt_instance[5]))
        self.maxpmtid = len(self.pmtmap)
        # self.c_in_LS = 2.99792458e2/1.578
        self.c_in_LS = 2.99792458e2


    def IdToPos(self, pmtid):
        return self.pmtmap[pmtid]

    def GetTOF(self, pmtid, event_vertex:np.ndarray):
        R = np.sum( (self.IdToPos(pmtid)[1:4]-event_vertex)**2 )**0.5
        return R/self.c_in_LS

    # Build a dictionary of the PMT location with theta and phi.
    def CalcDict(self):
        thetas = []
        thetaphi_dict = {}
        thetaphis = []
        for key in self.pmtmap:
            (pmtid, x, y, z, theta, phi) = self.pmtmap[key]
            if theta not in thetas:
                thetas.append(theta)
            thetaphis.append((theta, phi))
        for theta in thetas:
            thetaphi_dict[str(theta)] = []
        for (theta, phi) in thetaphis:
            thetaphi_dict[str(theta)].append(phi)
        for key in thetaphi_dict:
            thetaphi_dict[key] = np.sort(thetaphi_dict[key])
        self.thetaphi_dict = thetaphi_dict
        self.thetas = np.sort(thetas)

    def CalcThetaPhiGrid(self):
        thetas = set()
        phis = set()
        for key in self.pmtmap:
            (pmtid, x, y, z, theta, phi) = self.pmtmap[key]
            thetas.add(theta)
            phis.add(phi)
        thetas = (np.array(list(thetas)) - 90.) * np.pi / 180.
        phis = (np.array(list(phis)) - 180.) * np.pi / 180.
        self.thetas = np.sort(thetas)
        self.phis = np.sort(phis)
        print(f"len(thetas) : {len(self.thetas)}, len(phis) : {len(self.phis)}")
        print(f"thetas -- max: {np.max(self.thetas)}, min: {np.min(self.thetas)}")
        print(f"phis   -- max: {np.max(self.phis)}, min: {np.min(self.phis)}")

    def CalcThetaPhiPmtPoints(self):
        self.points_thetaphi = []  # NOT using set here is because we believe that in the pmt map file , there is no duplicate points
        self.thetas_set = set()
        for key in self.pmtmap:
            (pmtid, x, y, z, theta, phi) = self.pmtmap[key]
            # theta = (theta - 90.) * np.pi / 180.
            # phi   = (phi - 180.) * np.pi / 180.
            theta = theta * np.pi / 180.
            phi   = phi  * np.pi / 180.
            self.points_thetaphi.append((theta, phi))
            self.thetas_set.add(theta)
        self.thetas_set = np.array(list(self.thetas_set))
        self.points_thetaphi = np.array(list(self.points_thetaphi))
        self.thetas = self.points_thetaphi[:, 0]
        self.phis = self.points_thetaphi[:, 1]
        # Get PMTids in the border of theta-phi planar

        self.pmtid_border_right = np.zeros((len(self.thetas_set)), dtype=np.int)  # because theta of pmts is aligned(map to self.thetas_set which has no depulicate theta)
        self.pmtid_border_left = np.zeros((len(self.thetas_set)), dtype=np.int)
        for i, theta in enumerate(self.thetas_set):
            indices_SameTheta = np.where(self.points_thetaphi[:, 0] == theta)[0]
            i_max_phi_SameTheta = indices_SameTheta[np.argmax(self.points_thetaphi[indices_SameTheta, 1])]
            i_min_phi_SameTheta = indices_SameTheta[np.argmin(self.points_thetaphi[indices_SameTheta, 1])]
            self.pmtid_border_left[i] = i_min_phi_SameTheta
            self.pmtid_border_right[i]= i_max_phi_SameTheta
        # print(self.thetas_set, "max index :", self.pmtid_border_right, "min index : ", self.pmtid_border_left)

    def CalcThetaPhiSquareGrid(self, n_grid: int):
        self.thetas = np.linspace(-np.pi * 0.5, np.pi * 0.5, n_grid)
        self.phis = np.linspace(-np.pi, np.pi, n_grid)

    def CalcBin(self, pmtid):
        if pmtid > self.maxpmtid:
            print('Wrong PMT ID')
            return (0, 0)
        (pmtid, x, y, z, theta, phi) = self.pmtmap[str(pmtid)]
        xbin = int(phi * 128. / 360)
        ybin = int(theta * 128. / 180)
        # print(pmtid, x, y, z, theta, phi, xbin, ybin)
        return (xbin, ybin)

    def CalcBin_ThetaPhiImage(self, pmtid):
        if pmtid > self.maxpmtid:
            print('Wrong PMT ID')
            return (0, 0)
        (pmtid, x, y, z, theta, phi) = self.pmtmap[str(pmtid)]
        # Using xbin and ybin, PMTs can be mapped into a image, which like a oval
        xbin = np.where(self.thetas == theta)[0]
        # xbin = np.where(self.thetaphi_dict[str(theta)] == phi)[0] + 112 - int(len(self.thetaphi_dict[str(theta)])/2)# When the theta is close to pi/2, we got the max length of phis
        ybin = np.where(self.phis == phi)[0]
        # print((xbin, ybin))
        return (xbin, ybin)

def CorrTOF(v_hittime:np.ndarray, v_pmt_pos:np.ndarray, evt_vertex:np.ndarray )->np.ndarray:
    check_TOF = False
    c = 2.99792458e2/1.578 # mm/ns
    v_dxyz = np.array([pmt_pos-evt_vertex for pmt_pos in v_pmt_pos])
    v_R = np.sqrt(np.sum(v_dxyz**2, axis=1))
    if check_TOF:
        print("################")
        print("TOF:\t", v_R/c)
        print("PMT position:\t", v_pmt_pos)
        print("vertex:\t", evt_vertex)
        print("v_R:\t", v_R)
    v_time_corr_TOF = v_hittime-v_R/c
    return v_time_corr_TOF

def CorrTOFByPMTID(v_hittime:np.ndarray, v_pmt_id:np.ndarray, evt_vertex:np.ndarray , pmtmap:PMTIDMap)->np.ndarray:
    index_large_pmts = v_pmt_id<pmtmap.maxpmtid
    v_TOF = np.array([pmtmap.GetTOF(pmtid, evt_vertex) for pmtid in v_pmt_id[index_large_pmts] ])
    v_time_corr_TOF = v_hittime[index_large_pmts]-v_TOF
    return v_time_corr_TOF

def xyz2latlong(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    long = np.arctan2(y, x)  # longitude as same as phi
    xy2 = x ** 2 + y ** 2
    lat = np.arctan2(z, np.sqrt(xy2))  # latitude as same as theta
    return lat, long


def PlotRawSignal(event2dimage, x, y, z):
    fig_hittime = plt.figure("hittime")
    ax = fig_hittime.add_subplot(111, projection='3d')
    indices = (event2dimage[1] != 0.)
    # print(indices)
    img_hittime = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage[1][indices], cmap=plt.hot(), s=1)
    # img_hittime = ax.scatter(x, y, z, c=event2dimage[1], cmap=plt.hot(), s=1)
    fig_hittime.colorbar(img_hittime)

    fig_eqen = plt.figure("eqen")
    ax = fig_eqen.add_subplot(111, projection='3d')
    indices = (event2dimage[0] != 0)
    img_eqen = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage[0][indices], cmap=plt.hot(), s=1)
    # img_eqen = ax.scatter(x, y, z, c=event2dimage[0], cmap=plt.hot(), s=1)
    fig_eqen.colorbar(img_eqen)


def PlotIntepSignal(event2dimage_intep, x, y, z):
    fig_hittime = plt.figure("hittime_intep")
    ax = fig_hittime.add_subplot(111, projection='3d')
    # indices = (event2dimage_intep[1] != 0.)
    # img_hittime = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage_intep[1][indices], cmap=plt.hot(), s=1)
    img_hittime = ax.scatter(x, y, z, c=event2dimage_intep[1], cmap=plt.hot(), s=1)
    ax.set_xlabel("Detector X")
    ax.set_ylabel("Detector Y")
    ax.set_zlabel("Detector Z")
    fig_hittime.colorbar(img_hittime)

    fig_eqen = plt.figure("eqen_intep")
    ax = fig_eqen.add_subplot(111, projection='3d')
    # indices = (event2dimage_intep[0] != 0)
    # img_eqen = ax.scatter(x[indices], y[indices], z[indices], c=event2dimage_intep[0][indices], cmap=plt.hot(), s=1)
    img_eqen = ax.scatter(x, y, z, c=event2dimage_intep[0], cmap=plt.hot(), s=1)
    ax.set_xlabel("Detector X")
    ax.set_ylabel("Detector Y")
    ax.set_zlabel("Detector Z")
    fig_eqen.colorbar(img_eqen)

def PlotSigPlanar(thetas, phis, sig_r2, name_fig:str="sig_planar"):
    fig_planar = plt.figure(name_fig)
    ax = fig_planar.add_subplot(111)
    img_planar = ax.scatter(phis, thetas, c=sig_r2, cmap=plt.hot(), s=1)
    plt.xlabel("phi")
    plt.ylabel("theta")
    fig_planar.colorbar(img_planar)

def Expand2dPlanarBySphere(thetas, phis, sig_r2:np.ndarray, pmtmap:PMTIDMap):
    sig_r2_expand = sig_r2.copy()
    thetas_expand = thetas.copy()
    phis_expand   = phis.copy()
    plot_expanded_planar:bool = False

    # PlotSigPlanar(thetas, phis, sig_r2, name_fig="raw_sig")
    # sig_r2_expand = np.concatenate((sig_r2_expand, sig_r2[pmtmap.pmtid_border_right]))
    # thetas_expand = np.concatenate((thetas_expand, pmtmap.thetas_set))
    # phis_expand   = np.concatenate((phis_expand  , phis[pmtmap.pmtid_border_right]-2*np.pi))
    #
    # sig_r2_expand = np.concatenate((sig_r2_expand, sig_r2[pmtmap.pmtid_border_left]))
    # thetas_expand = np.concatenate((thetas_expand, pmtmap.thetas_set))
    # phis_expand   = np.concatenate((phis_expand  , phis[pmtmap.pmtid_border_left]+2*np.pi))
    # PlotSigPlanar(thetas_expand, phis_expand, sig_r2_expand, name_fig="add left edge")

    sig_r2_expand = np.concatenate((sig_r2_expand, sig_r2))
    thetas_expand = np.concatenate((thetas_expand, thetas))
    phis_expand   = np.concatenate((phis_expand  , phis-2*np.pi))

    sig_r2_expand = np.concatenate((sig_r2_expand, sig_r2))
    thetas_expand = np.concatenate((thetas_expand, thetas))
    phis_expand   = np.concatenate((phis_expand  , phis+2*np.pi))
    if plot_expanded_planar:
        PlotSigPlanar(thetas, phis, sig_r2, name_fig="raw_sig")
        PlotSigPlanar(thetas_expand, phis_expand, sig_r2_expand, name_fig="expanded sig")
        plt.show()
        exit()
    return (thetas_expand, phis_expand, sig_r2_expand)


def interp_pmt2mesh(sig_r2, thetas, phis, V, pmtmap, method="linear", do_calcgrid=False, dtype=np.float32):
    ele, azi = xyz2latlong(V)
    check_interp_range:bool = False
    plot_planar_afterInterp = False
    if check_interp_range:
        print("###########Checking whether raw data range is matched with the interplotation range############")
        print(f"ele range: {np.min(ele)}--{np.max(ele)}")
        print(f"azi range: {np.min(azi)}--{np.max(azi)}")
        print(f"thetas -- max: {np.max(thetas)}, min: {np.min(thetas)}")
        print(f"phis   -- max: {np.max(phis)}, min: {np.min(phis)}")
        s2 = np.array([ele, azi]).T
        print("s2:   ", s2.shape)
        print("sig_r2:   ", sig_r2.shape)
        print("(theta, phis):  ", (thetas[:10], phis[:10]))
        print("###############################################################################################")
    (thetas, phis, sig_r2) = Expand2dPlanarBySphere(thetas, phis, sig_r2, pmtmap)
    # sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1) #aims to add sig_r2 images one column
    if do_calcgrid:
        intp = RegularGridInterpolator((thetas, phis), sig_r2, method=method)
        sig_s2 = intp(s2).astype(dtype)
    else:
        indices = (sig_r2 != 0.)
        sig_s2 = griddata((thetas[indices], phis[indices]), sig_r2[indices], (ele, azi), method=method, fill_value=0.)
    # print("sig_s2 : ", sig_s2, "shape: ", sig_s2.shape)  # sig_s2 :  (642,)
    if plot_planar_afterInterp:
        fig_planar = plt.figure("planar after interp")
        ax = fig_planar.add_subplot(111)
        img = ax.scatter(azi, ele, c=sig_s2, cmap=plt.hot(), s=1)
        fig_planar.colorbar(img)
        plt.xlabel("phi")
        plt.ylabel("theta")
        # plt.show()
        # exit()
    return sig_s2

def GetOneEventImage(pmtids:np.ndarray, hittime:np.ndarray, npes:np.ndarray, pmtmap:PMTIDMap, V,
                     do_calcgrid:bool=False, max_n_points_grid:bool=True, subtract_TOF=False,
                     event_vertex:np.ndarray=(0,0,0)):
    if do_calcgrid == False:
        event2dimg = np.zeros((2, len(pmtmap.thetas)), dtype=np.float32)
    else:
        event2dimg = np.zeros((2, len(pmtmap.thetas), len(pmtmap.phis)), dtype=np.float32)

    # event2dimg = np.zeros((2, n_grid, n_grid), dtype=np.float16)
    event2dimg_interp = np.zeros((2, len(V)), dtype=np.float32)
    for j in range(len(pmtids)):
        if pmtids[j] >= 17612:
            continue
        # delta_t = (pmtmap.GetTOF(pmtid=pmtids[j], event_vertex=event_vertex) if subtract_TOF else 0.)
        delta_t = 0.
        if hittime[j] - delta_t < 0:
            continue

        if max_n_points_grid:
            if do_calcgrid:
                (xbin, ybin) = pmtmap.CalcBin_ThetaPhiImage(pmtids[j])
            else:
                i_pmt = pmtids[j]
        else:
            (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
        # if ybin>124:
        #     print(pmtids[i][j])
        if do_calcgrid == False:
            event2dimg[0, i_pmt] += npes[j]
            # if event2dimg[1, xbin, ybin] < 0.001 and event2dimg[1, xbin, ybin]>-0.001:
            if event2dimg[1, i_pmt] == 0:
                event2dimg[1, i_pmt] = hittime[j] - delta_t
            else:
                event2dimg[1, i_pmt] = min(hittime[j]-delta_t, event2dimg[1, i_pmt])
        else:
            event2dimg[0, xbin, ybin] += npes[j]
            # if event2dimg[1, xbin, ybin] < 0.001 and event2dimg[1, xbin, ybin]>-0.001:
            if event2dimg[1, xbin, ybin] == 0:
                event2dimg[1, xbin, ybin] = hittime[j]-delta_t
            else:
                event2dimg[1, xbin, ybin] = min(hittime[j]-delta_t, event2dimg[1, xbin, ybin])
    try:
        event2dimg_interp[0] = interp_pmt2mesh(event2dimg[0], pmtmap.thetas, pmtmap.phis, V, pmtmap, method="linear")
        event2dimg_interp[1] = interp_pmt2mesh(event2dimg[1], pmtmap.thetas, pmtmap.phis, V, pmtmap, method="nearest")
    except:
        print("event2dimg is empty!!!! Continue")

    return (event2dimg, event2dimg_interp)
