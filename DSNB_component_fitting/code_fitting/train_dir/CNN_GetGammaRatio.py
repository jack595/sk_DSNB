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
            # nn.Dropout(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128*2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128*2, 128*3, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.Dropout(0.25)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(768, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 11),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(11, 1 )
        self.fc4 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        # print(x.shape)
        x = self.fc1(x)
        # output layer
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

class CNN1D_2(nn.Module):
    def __init__(self):
        super(CNN1D_2, self).__init__()
        n = 10
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=32,  # n_filter
                      kernel_size=9,  # filter size
                      stride=1,  # filter step
                      padding=4,  # con2d出来的图片大小不变
                      ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.BatchNorm1d(32)# 1x2采样，o

        )
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 9, 1, 4),
                                   nn.LeakyReLU(),
                                   nn.MaxPool1d(2),
                                   # nn.BatchNorm1d(64)
                                   )

        self.conv3 = nn.Sequential(nn.Conv1d(64, 128,  9, 1, 4),
                                   nn.LeakyReLU(),
                                   nn.MaxPool1d(2),
                                   # nn.BatchNorm1d(128),
                                    nn.Conv1d(128, 256,  9, 1, 4),
                                   nn.LeakyReLU(),
                                   nn.MaxPool1d(2),
                                   # nn.BatchNorm1d(256),
                                    nn.Conv1d( 256,512,  9, 1, 4),
                                   nn.LeakyReLU(),
                                   nn.MaxPool1d(2),
                                   # nn.BatchNorm1d(512)
                                   )
        n_out_input = 512 * 1 * 2
        self.out = nn.Sequential(nn.Linear(n_out_input, int(n_out_input/2)),
                                 nn.Linear(int(n_out_input/2), int(n_out_input/4)),
                                 nn.Linear(int(n_out_input/4), 1),
                                 nn.Sigmoid())
        # self.out2 = nn.Sigmoid()

    def forward(self, x):
        print_size =False
        # x = x.view(x.size(0), 1, 256)
        print(x.size()) if print_size else 0
        x = self.conv1(x)
        print(x.size()) if print_size else 0
        x = self.conv2(x)
        print(x.size()) if print_size else 0
        x = self.conv3(x)
        print(x.size()) if print_size else 0
        x = x.view(x.size(0), -1)
        x = self.out(x)
        # x = self.out2(x)
        return x