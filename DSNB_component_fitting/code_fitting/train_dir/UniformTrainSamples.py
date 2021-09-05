# -*- coding:utf-8 -*-
# @Time: 2021/7/13 21:52
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: UniformTrainSamples.py
import matplotlib.pylab as plt
import numpy as np
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

import sys
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
from LoadMultiFiles import LoadMultiFiles
class PrepareData:
    def __init__(self, name_files:str):
        self.dir_evts = LoadMultiFiles(name_files)
        print(self.dir_evts)

if __name__ == '__main__':
    prepare_data = PrepareData("./predict_withpdgdep/predict_*.npz")

