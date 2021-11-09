import numpy as np
n_bkg = 0
# for i in range(134):
for i in [1]:
    # with np.load(f"./data_three_components/{i}.npz", allow_pickle=True) as f:
    with np.load(f"./data/{i}.npz", allow_pickle=True) as f:
        print(f"{i}.npz:\t")
        print(f.files)
        print("length of signal:\t",len(f["sig_NoWeightE"]))
        print("length of background:\t",len(f["bkg_NoWeightE"]))
        print("bkg_NoWeightE:\t", f["bkg_NoWeightE"][-5:])
        n_bkg += len(f["bkg_NoWeightE"])
        print("###########################################")
print("Total:\t", n_bkg)


