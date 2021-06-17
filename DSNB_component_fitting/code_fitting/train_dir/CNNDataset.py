import matplotlib.pylab as plt
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader,Dataset
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
class CNNDataset(Dataset):
    def __init__(self, v_time_profile, v_gamma_ratio):
        self.v_time_profiles = v_time_profile
        self.v_gamma_ratio = v_gamma_ratio
    def __len__(self):
        return len(self.v_time_profiles)
    def __getitem__(self, idx):
        time_profile = np.array(self.v_time_profiles[idx]).reshape((1,-1))
        gamma_ratio = np.array(self.v_gamma_ratio[idx])
        return (torch.from_numpy(time_profile).float(), torch.from_numpy(gamma_ratio).float())

