import matplotlib.pylab as plt
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import numpy as np

plt.figure()
label_version = "v1_TMVA"
name_input_dir = "./fit_result_npz/"
name_file_2d = name_input_dir+f"v_uplimit_2d_{label_version}.npy"
v_uplimit_2d = np.load(name_file_2d, allow_pickle=True)
name_file_1d = name_input_dir+f"v_uplimit_1d_{label_version}.npy"
v_uplimit_1d = np.load(name_file_1d, allow_pickle=True)
bins = np.linspace(0, 50, 10)
h_uplimit_1d = plt.hist(v_uplimit_1d, histtype="step", bins=bins, label="1d fitting")
h_uplimit_2d = plt.hist(v_uplimit_2d, histtype="step", bins=bins, label="2d fitting")
median_uplimit_1d = np.median((v_uplimit_1d))
median_uplimit_2d = np.median((v_uplimit_2d))
median_line_y = np.max(np.concatenate((h_uplimit_1d[0], h_uplimit_2d[0])))
# print("median:\t", median_uplimit)
plt.plot([median_uplimit_1d, median_uplimit_1d], [0, median_line_y], "--", label="1d median: {:.2f}".format(median_uplimit_1d))
plt.plot([median_uplimit_2d, median_uplimit_2d], [0, median_line_y], "--", label="2d median: {:.2f}".format(median_uplimit_2d))
plt.xlabel("Uplimit of Number of signal")
plt.legend()
plt.savefig(f"./figure_save/Uplimit_1dAnd2d_{label_version}.png")

"""
Substrate the upper limits between 2D fitting and 1D fitting 
"""
plt.figure()
plt.hist(v_uplimit_1d-v_uplimit_2d, bins=50)
plt.xlabel("$Uplimit_{1D}-Uplimit_{2D}$")
plt.title("Comparison Upper Limits between 1D and 2D")
plt.savefig((f"./figure_save/Uplimit_Comparison_{label_version}.png"))


name_file_chi2_2d = name_input_dir+f"v_chi2_2d_{label_version}.npz"
name_file_chi2_1d = name_input_dir+f"v_chi2_1d_{label_version}.npz"
f_2d = np.load(name_file_chi2_2d, allow_pickle=True)
f_1d = np.load(name_file_chi2_1d, allow_pickle=True)
v_n_sig_2d = f_2d["v_n_sig"]
v_n_sig_1d = f_1d["v_n_sig"]
v_chi2_2d = f_2d["v_chi2"]
v_chi2_1d = f_1d["v_chi2"]

n_lines_chi2 = 10
chi2_criteria = 2.706
plt.figure()
for i in range(n_lines_chi2):
    plt.plot(v_n_sig_2d, v_chi2_2d[i])
plt.xlabel("$N_{Signal}$")
plt.title("2D Fitting ")
plt.plot([0, np.max(v_n_sig_2d)], [chi2_criteria, chi2_criteria], "--", label="90% confidence")
plt.legend()
plt.savefig(name_input_dir+f"chi2_profile_2d_10lines_{label_version}.png")

plt.figure()
for i in range(n_lines_chi2):
    plt.plot(v_n_sig_1d, v_chi2_1d[i])
plt.xlabel("$N_{Signal}$")
plt.title("1D Fitting ")
plt.plot([0, np.max(v_n_sig_2d)], [chi2_criteria, chi2_criteria], "--", label="90% confidence")
plt.legend()
plt.savefig(name_input_dir+f"chi2_profile_1d_10lines_{label_version}.png")

plt.show()
