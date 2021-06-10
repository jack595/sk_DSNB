# -*- coding:utf-8 -*-
# @Time: 2021/5/21 9:37
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: DatasetProtonDecay_uscnn.py
import matplotlib.pylab as plt
import numpy as np
import ROOT
# import uproot as up
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import pickle
import sys
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/")
from python_script.usgcnnTools import PMTIDMap, GetOneEventImage, PlotRawSignal, PlotIntepSignal
def GetugscnnData(mapfile, sig_dir, bkg_dir, outfile='', start_entries=0):
        # The csv file of PMT map must have the same tag as the MC production.
    plot_result_sig: bool = False
    plot_result_bkg: bool = True
    max_n_points_grid: bool = True
    do_calcgrid: bool = False
    hittime_cut = 100 #ns
    pmtmap = PMTIDMap(mapfile)
    file_mesh = "./mesh_files/icosphere_5.pkl"
    p = pickle.load(open(file_mesh, "rb"))
    V = p['V']
    # pmtmap.CalcDict()
    n_grid = 128
    if max_n_points_grid:
        if do_calcgrid:
            pmtmap.CalcThetaPhiGrid()
        else:
            pmtmap.CalcThetaPhiPmtPoints()
    else:
        pmtmap.CalcThetaPhiSquareGrid(n_grid)
    if plot_result_sig or plot_result_bkg:
        if max_n_points_grid:
            if do_calcgrid:
                PHIS, THETAS = np.meshgrid(pmtmap.phis,
                                           pmtmap.thetas)  # Attention !!! Here we must be aware of the order of two inputs!!
            else:
                PHIS, THETAS = pmtmap.phis, pmtmap.thetas
        else:
            PHIS, THETAS = np.meshgrid(pmtmap.phis,
                                       pmtmap.thetas)  # Attention !!! Here we must be aware of the order of two inputs!!
            # print(f"thetas:{pmtmap.thetas}")
            # print(f"grid(thetas): {THETAS}")
        x_raw_grid = np.cos(THETAS) * np.cos(PHIS)
        y_raw_grid = np.cos(THETAS) * np.sin(PHIS)
        z_raw_grid = np.sin(THETAS)
        x_V, y_V, z_V = V[:, 0], V[:, 1], V[:, 2]

    bkgchain = ROOT.TChain('simevent')
    sigchain = ROOT.TChain('simevent')

    if focus_mode == "sig" or focus_mode=="":
        sigchain.Add(sig_dir)
    if focus_mode == "bkg" or focus_mode=="":
        bkgchain.Add(bkg_dir)

    print("Load Raw Data Successfully!!")
    pmtinfos = []
    types = []
    eqen_batch = []
    vertices = []
    if focus_mode == "":
        batchsize = bkgchain.GetEntries()  # because the entries in bkg file is fewer than in sig files ,so we set the batch size as entries contained in one bkg file
    else:
        batchsize = 20  # because the entries in bkg file is fewer than in sig files ,so we set the batch size as entries contained in one bkg file
    # print("batchsize:  ",bkgchain.GetEntries())
    # batchsize = 270
    for batchentry in range(batchsize):
        # save charge and hittime to 3D array
        if focus_mode == "sig" or focus_mode=="":
            i_sig = start_entries + batchentry
            if batchentry % 10 == 0:
                print("processing batchentry : ", batchentry)
            if focus_mode == "":
                if i_sig >= sigchain.GetEntries() or batchentry > bkgchain.GetEntries():
                    print("continued i_sig: ",i_sig)
                    continue
            sigchain.GetEntry(i_sig)
            pmtids = np.array(sigchain.t_pmtid, dtype=np.int32)
            npes = np.array(sigchain.t_npe, dtype=np.float32)
            hittime = np.array(sigchain.t_hittime, dtype=np.float32)
            eqen = sigchain.t_Qedep

            index_sig_hittime_cut = hittime<hittime_cut
            (event2dimg, event2dimg_interp) = GetOneEventImage(pmtids[index_sig_hittime_cut], hittime[index_sig_hittime_cut], npes[index_sig_hittime_cut], pmtmap, V, do_calcgrid, max_n_points_grid)
            if plot_result_sig:
                PlotRawSignal(event2dimg, x_raw_grid, y_raw_grid, z_raw_grid)
                PlotIntepSignal(event2dimg_interp, x_V, y_V, z_V)
                plt.show()
                exit()
            pmtinfos.append(event2dimg_interp)
            types.append(1)
            eqen_batch.append(eqen)
            vertices.append([sigchain.t_QedepX, sigchain.t_QedepY, sigchain.t_QedepZ])

        if focus_mode == "bkg" or focus_mode=="":
            bkgchain.GetEntry(batchentry)
            pmtids = np.array(bkgchain.t_pmtid, dtype=np.int32)
            npes = np.array(bkgchain.t_npe, dtype=np.float32)
            hittime = np.array(bkgchain.t_hittime, dtype=np.float32)
            eqen = bkgchain.t_Qedep

            index_bkg_hittime_cut = hittime<hittime_cut
            (event2dimg, event2dimg_interp) = GetOneEventImage(pmtids[index_bkg_hittime_cut], hittime[index_bkg_hittime_cut], npes[index_bkg_hittime_cut], pmtmap, V, do_calcgrid, max_n_points_grid)
            if plot_result_bkg:
                PlotRawSignal(event2dimg, x_raw_grid, y_raw_grid, z_raw_grid)
                PlotIntepSignal(event2dimg_interp, x_V, y_V, z_V)
                plt.show()
                exit()
            pmtinfos.append(event2dimg_interp)
            types.append(0)
            vertices.append([bkgchain.t_QedepX, bkgchain.t_QedepY, bkgchain.t_QedepZ])
            eqen_batch.append(eqen)
            # print("len(pmtinfos):  ",len(pmtinfos))

    indices = np.arange(len(pmtinfos))
    np.random.shuffle(indices)
    pmtinfos = np.array(pmtinfos)[indices]
    types = np.array(types, dtype=np.int32)[indices]
    eqen_batch = np.array(eqen_batch)[indices]
    vertices = np.array(vertices)[indices]
    print("types After shuffle:   ", types)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='JUNO ML dataset builder.')
    parser.add_argument('--pmtmap', type=str, help='csc file of PMT map in JUNO.',
    default="/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre0/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv")
    parser.add_argument('--sigdir', '-s', type=str, help='Input root file(Signal).',
                        default="/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/Atm_hu/data1/Uedm_1572.root")
    parser.add_argument('--bkgdir', '-b', type=str, help='Input root file(Background).',
                        default="/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/Atm_hu/data1/Uedm_1571.root")
    parser.add_argument('--outfile', '-o', type=str, help='Output root file.', default="./try.npz")
    parser.add_argument('--StartEntries', '-e', type=int, help='Start Entry of sig_file.', default=0)
    parser.add_argument('--focus', '-f', type=str, help="focus mode(only study sig or bkg)", default="bkg") #“”is to turn off the focus mode
    args = parser.parse_args()

    focus_mode = args.focus
    GetugscnnData(args.pmtmap, args.sigdir, args.bkgdir, args.outfile, args.StartEntries)