# -*- coding:utf-8 -*-
# @Time: 2021/6/1 15:06
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: NumpyTools.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
def ReBin(v:np.ndarray):
    length_v = len(v)
    v_odd = v[0:length_v:2]
    v_even = v[1:length_v:2]
    print(v_odd)
    print(v_even)

def AlignRisingEdge(v_to_align:np.ndarray, threshold:float, i_loc_aligned=10):
    i_over_threshold = np.where(v_to_align>threshold)[0][0]-i_loc_aligned
    if i_over_threshold>=len(v_to_align):
        print(f"ERROR!:Cannot get the rising edge using threshold ({threshold})")
        exit()
    v_return = v_to_align
    if i_over_threshold>0:
        v_return = np.delete(v_to_align, np.arange(i_over_threshold))
        v_return = np.concatenate((v_return, np.zeros(i_over_threshold)))
    elif i_over_threshold < 0:
        i_append = -i_over_threshold
        v_return = np.concatenate((np.zeros(i_append), v_to_align))[:len(v_to_align)]
    return v_return

if __name__ == '__main__':
    # Test AlignRisingEdge
    h_time = np.array([0,0,0,1,3,5,6,7,15,20,16,14,12,10,9,7,5,0,0])
    print(AlignRisingEdge(h_time, threshold=5, i_loc_aligned=10))
