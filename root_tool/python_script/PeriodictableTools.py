# -*- coding:utf-8 -*-
# @Time: 2022/2/19 9:53
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PeriodictableTools.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys
import periodictable as pt
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")

class PeriodictableTools:
    def __init__(self):
        self.list_elements = [str(element) for element in pt.elements]
        self.list_charge = range(len(self.list_elements))
        self.map_charge = {element:charge for element, charge in zip(self.list_elements, self.list_charge)}
    def MapToCharge(self, symbol:str):
        return self.map_charge[symbol]

if __name__ == "__main__":
    table_tool = PeriodictableTools()
    print(table_tool.MapToCharge("Na"))