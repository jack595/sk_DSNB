# -*- coding:utf-8 -*-
# @Time: 2021/10/11 20:13
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PSD_dataset.py
import os.path

import matplotlib.pylab as plt
import numpy as np
import ROOT
import argparse

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

def GetBins():
    # bins_hist = [-19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
    #              7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
    #              48, 50, 52, 54, 56, 58, 60, 62, 66, 72, 80, 90, 102, 116, 132, 150, 170, 192, 216, 242, 270, 300, 332, 366,
    #              402, 440, 480, 522, 566, 612, 660, 710, 762, 816]
    # bins_hist = list(range(-19,20,3)) + list(range(20, 62,6)) + [62, 72, 80, 90, 102, 116, 132, 150, 170, 192, 216, 242, 270, 300, 332, 366,
    #              402, 440, 480, 522, 566, 612, 660, 710, 762, 816]
    # bins_hist = list(range(-20,20,3)) + list(range(20, 63,6)) + [ 72, 80, 90, 102, 116, 132, 150, 170, 192, 216, 242, 270, 300, 332, 366,
    #              402, 440, 480, 522, 566, 612, 660, 710, 762, 816]
    bins_hist = list(range(-20,20,3)) + list(range(22, 66,6))  + [ 72, 80, 90, 102, 116, 132, 150, 170, 192, 216, 242, 270, 300, 332, 366,
                                                                  402, 440, 480, 522, 566, 612, 660, 710, 762, 816]
    return bins_hist

class PSDDataset:
    def __init__(self, name_file:str, tag:str ,key_chain:str="psdtree"):
        self.tag = tag

        self.dir_events = {"h_time":[], "h_time_with_charge":[], "vertex":[],"R3":[], "equen":[],
                           "edep":[], "tag":[]}

        self.bins = np.array(GetBins())
        self.bins_width = np.diff(self.bins)
        self.bins_center = (self.bins[:-1]+self.bins[1:])/2

        self.chain = ROOT.TChain(key_chain)
        self.chain.Add(name_file)
        self.entries = self.chain.GetEntries()
        print(f"Entries of Events:\t", self.entries)

        # Option for debugging
        self.debug_plot_mean_profile = True
        self.debug_plot_profile = True
    def SetBins(self, bins):
        """
        Set bin strategy so we can do debugging
        :return:
        """
        self.bins = np.array(bins)
        self.bins_width = np.diff(self.bins)
        self.bins_center = (self.bins[:-1]+self.bins[1:])/2

    def DrawTimeProfile(self,axes:plt.axes=None, plot_mean=True, n_lines_to_plot=20, color="blue",
                        label:str="", log=False):
        if isinstance(axes,list):
            fig, axes = plt.subplots(1, 2, figsize=(16,8))
        if plot_mean:
            axes[0].plot(self.bins_center, np.mean(self.dir_events["h_time"]/self.bins_width, axis=0), label=label)
            axes[1].plot(self.bins_center, np.mean(self.dir_events["h_time_with_charge"]/self.bins_width, axis=0), label=label)
        else:
            for i in range(n_lines_to_plot):
                if i == 0:
                    axes[0].plot(self.bins_center, self.dir_events["h_time"][i]/self.bins_width, color=color, linewidth=1,label=label)
                    axes[1].plot(self.bins_center, self.dir_events["h_time_with_charge"][i]/self.bins_width, linewidth=1,color=color,label=label)
                else:
                    axes[0].plot(self.bins_center, self.dir_events["h_time"][i]/self.bins_width, color=color, linewidth=1)
                    axes[1].plot(self.bins_center, self.dir_events["h_time_with_charge"][i]/self.bins_width, linewidth=1,color=color)

        axes[0].set_title("w/o charge")
        axes[1].set_title("w/  charge")
        for ax in axes:
            if log:
                ax.semilogy()
            ax.set_xlabel("Time [ ns ]")
            ax.legend()


    def LoadData(self, equen_downlimit=None, equen_uplimit=None):
        print(f"Bins:\t", self.bins)
        for i_entry in range(self.entries):
            if i_entry%200 == 0:
                print(f"Processing i_entry = {i_entry}")
            self.chain.GetEntry(i_entry)

            # Apply Energy Cut
            if (equen_uplimit != None) or (equen_downlimit!=None):
                if (self.chain.Eqen<equen_downlimit) or (self.chain.Eqen>equen_uplimit):
                    continue

            # Save events information
            self.dir_events["equen"].append(self.chain.Eqen)
            self.dir_events["vertex"].append(np.array([self.chain.X, self.chain.Y, self.chain.Z]))
            self.dir_events["R3"].append( ( np.sqrt(self.chain.X**2+self.chain.Y**2+self.chain.Z**2)/1000. ) **3)
            self.dir_events["edep"].append(self.chain.Edep)
            self.dir_events["tag"].append(self.tag)

            charge = np.array(self.chain.Charge)
            time = np.array(self.chain.Time)

            h_time,_ = np.histogram(time, bins=self.bins)
            h_time_with_charge,_ = np.histogram(time, bins=self.bins, weights=charge)
            h_time = h_time/np.max(h_time)
            h_time_with_charge = h_time_with_charge/np.max(h_time_with_charge)

            self.dir_events["h_time"].append(np.array(h_time))
            self.dir_events["h_time_with_charge"].append(np.array(h_time_with_charge))

    def SaveDataset(self, name_file_to_save):
        path_dir_save = os.path.dirname(name_file_to_save)
        print(path_dir_save)
        if not os.path.isdir(path_dir_save):
            os.makedirs(path_dir_save)

        np.savez(name_file_to_save, dir_events=self.dir_events)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='sklearn dataset builder.')
    parser.add_argument("--input_file", "-i", type=str, default="root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/PSD_LowE/alpha/RawData/detsimNB_1.root",
                        help="name of input file")
    parser.add_argument("--output_file", "-o", type=str, default="alpha_0.npz", help="name of output file")
    parser.add_argument("--tag", "-t", type=str, default="alpha", help="events tag")
    parser.add_argument("--debug", "-d", action="store_true",help="whether plot figures to debug" )
    arg = parser.parse_args()

    psd_data = PSDDataset(arg.input_file, arg.tag)
    psd_data.LoadData()
    
    if arg.debug:
        psd_data.DrawTimeProfile(plot_mean=False)
        plt.show()

    print(f"Saving {arg.output_file}........")
    psd_data.SaveDataset(arg.output_file)





