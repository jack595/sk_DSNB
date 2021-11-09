import matplotlib.pylab as plt
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import numpy as np
sensitivity_10yrs = {}
label_version = "v1_TMVA"
label_C11_cut = "_add_tccut"
label_add_FV2 = "_add_FV2"
v_options = ["", label_C11_cut, label_C11_cut+label_add_FV2]
v_colors = ["b", "g", "m"]
v_legends = [" (FV1 w/o TC-Cut)"," (FV1 w/ TC-Cut)", " (FV1+FV2 w/ TC-Cut)" ]
for i, key in enumerate(v_options):
    for label_fit_method in ["1d", "2d"]:
        f = np.load(f"./fit_result_npz/discover_potential_{label_fit_method}_{label_version}{key}.npz", allow_pickle=True)
        v_year = f["v_year"]
        v_sensitivity = f["v_sensitivity"]
        if label_fit_method =="1d":
            plt.plot(v_year, v_sensitivity, label=label_fit_method+f" fitting{v_legends[i]}", color=v_colors[i], ls="--")
        elif label_fit_method == "2d":
            plt.plot(v_year, v_sensitivity, label=label_fit_method+f" fitting{v_legends[i]}", color=v_colors[i] )
        sensitivity_10yrs[label_fit_method+v_legends[i]] = v_sensitivity[-1]
plt.legend()
plt.ylim(0,5.5)
plt.title("Discovery Potential for DSNB")
# plt.xlabel("exposure [ 14.7 kt $*$ yr ] ")
plt.xlabel("Time [ year ]")
plt.ylabel("Sensitivity[$\sigma$]")
for i, key in enumerate(v_legends):
    print("############# Ten Years Sensitivity ##############")
    print(f"1D fitting{v_legends[i]}:\t", "{:.2f}".format(sensitivity_10yrs["1d"+v_legends[i]]))
    print(f"2D fitting{v_legends[i]}:\t", "{:.2f}".format(sensitivity_10yrs["2d"+v_legends[i]]))
    print("Sensitivity Improved:\t", "{:.2f} %".format(100*(sensitivity_10yrs["2d"+v_legends[i]]-sensitivity_10yrs["1d"+v_legends[i]])/sensitivity_10yrs["2d"+v_legends[i]]))
    print("##################################################")

f = np.load(f"./fit_result_npz/discover_potential_1d_{label_version}.npz", allow_pickle=True)
v_year = f["v_year"]
v_sensitivity = f["v_sensitivity"]
v_sensitivity_zhangyy = [1.511,2.035,2.396,2.675,2.904,3.097,3.266,3.414,3.547,3.668]
plt.figure()
# plt.plot(v_sensitivity, label="1d fitting(Simulation Samples) ")
# plt.plot(v_sensitivity_zhangyy, label="1d fitting(Energy Spectrum)")
plt.plot(v_sensitivity, label="1d fitting(Simulation Samples Eff.) ")
plt.plot(v_sensitivity_zhangyy, label="1d fitting(Spectrum Eff.)")
plt.ylim(0, 5.5)
plt.title("Discovery Potential for DSNB")
plt.xlabel("exposure [ 14.7 kt $*$ yr ] ")
plt.ylabel("Sensitivity[$\sigma$]")
plt.legend()

plt.show()

