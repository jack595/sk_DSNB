# -*- coding:utf-8 -*-
# @Time: 2021/2/20 15:52
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PmtIDMap.py
import numpy as np
import uproot as up
class PMTIDMap:
    n_pmt = 17612

    def __init__(self, mapfile:str="/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv"):
        """
        :param mapfile: file about pmts' information, csv file or root file to be input, when root file is input, idToIsHam() is enable.
        mapfile example: /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv
                         /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/data/Simulation/ElecSim/PmtData_Lpmt.root
        """
        self.pmt_xyz = np.zeros((self.n_pmt, 3))
        self.pmt_theta_phi = np.zeros((self.n_pmt, 2))
        self.pmt_isHam = np.zeros((self.n_pmt, 1))
        self.is_rootfile_input = None
        if "csv" in mapfile:
            self.pmtmap = np.loadtxt(mapfile)
            for i_pmtmap in self.pmtmap:
                self.pmt_xyz[int(i_pmtmap[0])] = i_pmtmap[1:4]
                self.pmt_theta_phi[int(i_pmtmap[0])] = i_pmtmap[-2:]
            self.is_rootfile_input = False
        elif "root" in mapfile:
            with up.open(mapfile) as f:
                tree_pmt = f["PmtData_Lpmt"]
                self.v_pmtid = np.array(tree_pmt["pmtID"], dtype=int)
                self.v_x_pmt = np.array(tree_pmt["pmtPosX"])
                self.v_y_pmt = np.array(tree_pmt["pmtPosY"])
                self.v_z_pmt = np.array(tree_pmt["pmtPosZ"])
                v_isHam = np.array(tree_pmt["MCP_Hama"], dtype=int)
                for i in range(len(self.v_pmtid)):
                    self.pmt_xyz[self.v_pmtid[i]] = np.array([self.v_x_pmt[i], self.v_y_pmt[i], self.v_z_pmt[i]])

                    self.pmt_isHam[self.v_pmtid[i]] = v_isHam[i]
            self.is_rootfile_input = True
        else:
            print("ERROR:\tcsv file or root file is supposed to be input!!!!")
            exit(1)
    def idToXYZ(self, pmtid:int):
        return np.array(self.pmt_xyz[pmtid])
    def idToThetaPhi(self, pmtid:int):
        if self.is_rootfile_input and np.all(self.pmt_theta_phi==0):
            for i in range(len(self.v_pmtid)):
                self.pmt_theta_phi[self.v_pmtid[i]] = np.array([np.rad2deg(np.arctan2(np.sqrt((self.v_x_pmt[i] ** 2 + self.v_y_pmt[i]**2)), self.v_z_pmt[i])),
                                                                np.rad2deg(np.arctan2(self.v_y_pmt[i], self.v_x_pmt[i]))])
        return np.array(self.pmt_theta_phi[pmtid])
    def idToIsHam(self, pmtid:int):
        if not self.is_rootfile_input:
            print("ERROR:\tNot root file input, idToIsHam() is unavailable!!!")
            exit(1)
        else:
            return np.array(self.pmt_isHam[pmtid])

if __name__ == '__main__':
    pmtmap = PMTIDMap("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv")
    print(pmtmap.pmt_xyz[-5:])
    print(pmtmap.pmt_theta_phi[-5:])
    print(pmtmap.idToXYZ(14))
    print(pmtmap.idToThetaPhi(14))
    print(pmtmap.idToThetaPhi(15))

    pmtmap_root = PMTIDMap("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre2/data/Simulation/ElecSim/PmtData_Lpmt.root")
    print(pmtmap_root.idToXYZ(14))
    print(pmtmap_root.idToThetaPhi(14))
    print(pmtmap_root.idToThetaPhi(15))




