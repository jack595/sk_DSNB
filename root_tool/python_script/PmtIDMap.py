# -*- coding:utf-8 -*-
# @Time: 2021/2/20 15:52
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PmtIDMap.py
import numpy as np
class PMTIDMap:
    def __init__(self, mapfile:str="/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv"):
        n_pmt = 17612
        self.pmtmap = np.loadtxt(mapfile)
        self.pmt_xyz = np.zeros((n_pmt, 3))
        self.pmt_theta_phi = np.zeros((n_pmt, 2))
        for i_pmtmap in self.pmtmap:
            self.pmt_xyz[int(i_pmtmap[0])] = i_pmtmap[1:4]
            self.pmt_theta_phi[int(i_pmtmap[0])] = i_pmtmap[-2:]
    def idToXYZ(self, pmtid:int):
        return np.array(self.pmt_xyz[pmtid])
    def idToThetaPhi(self, pmtid:int):
        return np.array(self.pmt_theta_phi[pmtid])

if __name__ == '__main__':
    pmtmap = PMTIDMap()
    print(pmtmap.pmt_xyz[-5:])
    print(pmtmap.pmt_theta_phi[-5:])
    print(pmtmap.idToXYZ(14))
    print(pmtmap.idToThetaPhi(14))




