# -*- coding:utf-8 -*-
# @Time: 2021/6/14 15:05
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: CNN_GetGammaRatio.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

from torch import nn
import torch

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 11),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(11, 1 )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # output layer
        x = self.fc2(x)
        x = self.fc3(x)

        return x
