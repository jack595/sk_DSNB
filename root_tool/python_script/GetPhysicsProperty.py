# -*- coding:utf-8 -*-
# @Time: 2021/6/4 15:38
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GetPhysicsProperty.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

from particle import Particle
class PDGMassMap:
    def __init__(self):
        self.list_base_pdg = np.concatenate( (np.arange(1,7),np.arange(11, 17),
                                             -np.arange(1,7), -np.arange(11, 17),
                                              np.array([22, 2212, 2112, 211, -211, 111]) ) )
        self.dir_pdg_mass_map = {}
        self.dir_pdg_name_map = {}
        self.dir_name_sub = {"p":"proton", "n":"neutron"}

    def GetBaseMass(self):
        """

        This Base directory is to save time not to search some repeat pdg
        :return:
        """
        self.mass_base_pdgs = []
        for pdg in self.list_base_pdg:
            p = Particle.from_pdgid(pdg)
            mass = p.mass
            if mass != None:
                self.mass_base_pdgs.append(mass)
            else:
                self.mass_base_pdgs.append(0.)
            self.dir_pdg_name_map[pdg] = p.name
        for i in range(len(self.mass_base_pdgs)):
            self.dir_pdg_mass_map[self.list_base_pdg[i]] = self.mass_base_pdgs[i]

    def PDGToMass(self, pdg):
        if len(self.dir_pdg_mass_map.keys()) == 0:
            self.GetBaseMass()
        if pdg in self.dir_pdg_mass_map.keys():
              return self.dir_pdg_mass_map[pdg]
        else:
            return Particle.from_pdgid(pdg).mass

    def PDGToName(self, pdg):
        if len(self.dir_pdg_name_map.keys()) == 0:
            self.GetBaseMass()
        if pdg in self.dir_pdg_name_map.keys():
            name_return = self.dir_pdg_name_map[pdg]
        else:
            name_return = Particle.from_pdgid(pdg).name
        if name_return in self.dir_name_sub.keys():
            return self.dir_name_sub[name_return]
        else:
            return name_return


def GetKineticE(momentum_square, mass):
    return np.sqrt(momentum_square+mass**2)-mass

def NameToPDGID(name_ion:str, delimiter="_"):
    """

    :param name_ion: the name of ion, for example He_4
    :return: PDGID
    """
    from PeriodictableTools import PeriodictableTools
    periodictable_tool = PeriodictableTools()
    Z = int(periodictable_tool.MapToCharge(name_ion.split(delimiter)[0]))
    N = int(name_ion.split(delimiter)[1])
    return int(N*10+Z*1e4+1e9)
