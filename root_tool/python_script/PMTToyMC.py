# -*- coding:utf-8 -*-
# @Time: 2022/2/12 11:43
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PMTToyMC.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys
import random
import csv

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

class PMTToyMC:
    def __init__(self, rate_dark_noise:float=7.0, TTS:float=1.6):
        self.rate_dark_noise = rate_dark_noise
        self.TTS = TTS
        self.time_template_waveform = []
        self.amplitude_template_waveform = []
        with open("/afs/ihep.ac.cn/users/l/luoxj/root_tool/Data/Waveform_Template_SPE_SPME.csv") as f:
            for row in csv.reader(f, delimiter=","):
                self.time_template_waveform.append(float(row[0]))
                self.amplitude_template_waveform.append(float(row[1]))
        plt.plot(self.time_template_waveform,self.amplitude_template_waveform )
        plt.show()
        exit()
        # self.SPE_template =
        
    def SetT(self, v_t ):
        self.v_t = np.array(v_t)
        
    def GenerateDarkNoise(self, win_extended=1000):
        t_min = min(self.v_t)-win_extended/2
        t_max = max(self.v_t)+win_extended/2
        win_t = t_max-t_min # ns
        n_dark_noise = np.random.poisson(win_t*self.rate_dark_noise*1e-9)
        print(win_t,n_dark_noise,win_t*self.rate_dark_noise*1e-9)
        return np.random.uniform(t_min, t_max, size=n_dark_noise)

    def AddTTS(self):
        v_smear = np.array([random.gauss(0, self.TTS) for _ in range(len(self.v_t))])
        self.v_t = self.v_t+v_smear

    def ToyMC(self, v_t, win_extended=1000):
        self.SetT(v_t)
        self.AddTTS()
        self.v_t = np.concatenate((self.v_t, self.GenerateDarkNoise(win_extended)))
        return self.v_t

    # def ConvolutionWithTemplate(self):
    #     self.
    
if __name__ == '__main__':
    toy_mc = PMTToyMC(0.1, 3)
    print(toy_mc.ToyMC(range(100)))
        
    