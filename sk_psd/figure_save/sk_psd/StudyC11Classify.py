#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pylab as plt
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import pandas as pd


# In[2]:


# load file
import numpy as np
# f = np.load("./model_maxtime_combine_jobs_DSNB_sk_data/predict_0.npz", allow_pickle=True)
f = np.load("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_time_jobs_DSNB_sk_data/predict_0.npz", allow_pickle=True)
print(f"key: {f.files}")
predict_proba = f["predict_proba"][:,1]
equen = f["equen"]
vertex = f["vertex"]
labels = f["labels"]
pdgs = f["pdg_bkg"]
# print("predict_proba: ", predict_proba)
# print("labels: ", labels)
# print("pdg:  ", pdgs)


# In[3]:


# seperate sig and bkg
dir_proba ={}
dir_vertex = {}
dir_equen = {}
dir_proba["sig"] = predict_proba[labels==1]
dir_proba["bkg"] = predict_proba[labels==0]
dir_vertex["sig"] = vertex[labels==1]
dir_vertex["bkg"] = vertex[labels==0]
dir_equen["sig"] = equen[labels==1]
dir_equen["bkg"] = equen[labels==0]

# print("Check bkg length: ")
# print("proba->  ", len(dir_proba["bkg"]))
# print("pdg-> ", len(pdgs))
# print("vertex-> ", len(dir_vertex["bkg"]))


# In[4]:


# Study pdgs
from collections import Counter
def GetNucleiNum(pdg_evt):
    n_nuclei = 0
    counter = Counter(pdg_evt)
    for key in counter:
        if key > 1000000000:
            n_nuclei += counter[key]
    return n_nuclei

def PdgToN(Nuclei_pdg):
    N = int(Nuclei_pdg/10)%1000
    Z = int(Nuclei_pdg/10000)%1000
    return (N, Z)

class OneNucleiEvts:
    def __init__(self):
        self.probs = []
        self.v_NZ = []
    def Print(self):
        print(f"probs: {self.probs}")
        print(f"v_NZ: {self.v_NZ}")
evt_1Nuclei = OneNucleiEvts()

v_pdg_multi_nuclei = []
v_proba_multi_nuclei = []
v_pdgs_one_nuclei = []
for i, pdg_evt in enumerate(pdgs):
    n_nuclei = GetNucleiNum(pdg_evt)
    if n_nuclei == 1 :
        Nuclei_pdg = pdg_evt[pdg_evt>1000000000][0]
        (N, Z ) = PdgToN(Nuclei_pdg)
        evt_1Nuclei.probs.append(dir_proba["bkg"][i])
        evt_1Nuclei.v_NZ.append([N,Z])
        v_pdgs_one_nuclei.append(pdg_evt)
    else:
        v_proba_multi_nuclei.append(dir_proba["bkg"][i] )
        v_pdg_multi_nuclei.append(pdg_evt)
# evt_1Nuclei.Print()


# ## Find The Right Criteria

# In[5]:


criteria_to_use = 0
for criteria in np.arange(0.9, 1, 0.0001 ):
    index_bkg_rightPredict = (dir_proba["bkg"]<criteria)
    counter_bkglike = Counter(index_bkg_rightPredict)
    eff_bkg = counter_bkglike[True]/len(index_bkg_rightPredict)

    index_sig_rightPredict = (dir_proba["sig"]>criteria)
    counter_sig_rightPredict = Counter(index_sig_rightPredict)
    eff_sig = counter_sig_rightPredict[True]/len(index_sig_rightPredict)
    if eff_bkg > 0.99:
        criteria_to_use = criteria
        print("Criteria :\t", criteria)
        print("Efficiency of bkg:\t", eff_bkg)
        print("Efficiency of sig:\t", eff_sig)
        print("###########################################")
        break
print("Will use criteria --> ", criteria_to_use)


# ## Check $^{11}C$ and $^{10}B$ Background (Signal like ratio)
# 

# In[6]:


def HistTimes(hist:np.ndarray, times:int):
    hist_return = list(hist)*times
    return np.array(hist_return)


# In[7]:


## Draw Nuclei distribution
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib.pylab as plt
criteria = criteria_to_use
index_siglike = (np.array(evt_1Nuclei.probs)>=criteria)
index_bkglike = (np.array(evt_1Nuclei.probs)<criteria)

labels = ["$H$", "$He$", "$Li$", "$Be$", "$B$", "$C$", "$N$", "$O$"]
plt.figure(figsize=(9, 6))
x =np.arange(1, len(labels)+1)
h_siglike = plt.hist(HistTimes(np.array(evt_1Nuclei.v_NZ)[index_siglike][:,1],10), bins=x , histtype='step', label="Sig-like*10($P_{sig}>=$"+"{:.2f}".format(criteria)+")")
h_bkglike = plt.hist(np.array(evt_1Nuclei.v_NZ)[index_bkglike][:,1], bins=x, histtype='step', label="Bkg-like($P_{sig}"+"<${:.2f}".format(criteria)+")")
plt.title("Background Distribution")
plt.xticks(x+0.5, labels)
plt.xlim([0, len(labels)+1])
# plt.xlabel("Z of Proton")
plt.legend(loc="upper left")

plt.figure(figsize=(9, 6))
plt.hist(HistTimes(np.array(evt_1Nuclei.v_NZ)[index_siglike][:,0],10), bins=range(0, 15), histtype='step', label="Sig-like*10($P_{sig}>=$"+"{:.2f}".format(criteria)+")")
plt.hist(np.array(evt_1Nuclei.v_NZ)[index_bkglike][:,0], bins=range(0, 15), histtype='step', label="Bkg-like($P_{sig}"+"<${:.2f}".format(criteria)+")")
plt.title("Background Distribution")
plt.xlabel("N of Nuclei")
plt.legend(loc="upper left")

plt.figure(figsize=(8, 5))
h_siglike = np.array(h_siglike[0])/10
h_bkglike = h_bkglike[0]
h_ratio = np.nan_to_num(np.array(h_siglike)/((np.array(h_bkglike))+np.array(h_siglike)))
plt.stem(h_ratio)
# plt.xlabel("Z of Proton")
plt.xticks(x-1, labels)
plt.ylabel("Ratio")
plt.title("Ratio(Sig-like/Total)")

# plt.show()


# In[8]:


print(h_siglike)
print(h_bkglike)

#handle multi-nuclei evt
# print(v_pdg_multi_nuclei, v_proba_multi_nuclei)
print(f"Total multi-nuclei : {len(v_proba_multi_nuclei)}")
v_proba_multi_nuclei = np.array(v_proba_multi_nuclei)
print(f"sig like : {len(v_proba_multi_nuclei[np.array(v_proba_multi_nuclei)>=criteria])}" )
print(f"bkg like : {len(v_proba_multi_nuclei[np.array(v_proba_multi_nuclei)<criteria])}" )
n_siglike_multi_nuclei = len(v_proba_multi_nuclei[np.array(v_proba_multi_nuclei)>=criteria])
n_bkglike_multi_nuclei = len(v_proba_multi_nuclei[np.array(v_proba_multi_nuclei)<criteria])
h_bkglike = np.append(h_bkglike, [n_bkglike_multi_nuclei])
h_siglike = np.append(h_siglike, [n_siglike_multi_nuclei])


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
Z = np.array(evt_1Nuclei.v_NZ)[:,1]
N = np.array(evt_1Nuclei.v_NZ)[:,0]
probs = np.array(evt_1Nuclei.probs)
from collections import Counter
counter_C = Counter(N[Z==6])

# h_C10 = plt.hist(probs[(Z==6) & (N==10)], bins=np.arange(0, 1, 0.05))
# plt.xlabel("PSD ouput")

print("Carbon:", counter_C)
print(np.array(v_pdgs_one_nuclei)[(Z==6) & (N==10)])
# plt.hist2d(N, Z, cmap=plt.hot())
# import pandas as pd
# pd.DataFrame(h_C10[1][:-1], h_C10[0])
counter_C11=Counter(probs[(Z==6) & (N==11)]>0.95)
counter_C10=Counter(probs[(Z==6) & (N==10)]>0.95)
counter_C9 =Counter(probs[(Z==6) & (N==9)]>0.95)
print("C11", counter_C11)
print("C10", counter_C10)
print("C9 ", counter_C9)


# In[10]:


C11_siglike = counter_C11[True]
C11_bkglike = counter_C11[False]
C10_siglike = counter_C10[True]
C10_bkglike = counter_C10[False]
C9_siglike = counter_C9[True]
C9_bkglike = counter_C9[False]
h_C_bkglike = [C9_bkglike, C10_bkglike, C11_bkglike]
h_C_siglike = [C9_siglike, C10_siglike, C11_siglike]
pd.DataFrame([h_C_bkglike, h_C_siglike ],
             index=["Background Like","Signal Like"], columns=["$^{9}C$","$^{10}C$","$^{11}C$"])


# In[11]:


index_C = np.where(np.array(labels)=="$C$")[0][0]
pd.DataFrame([np.array(h_bkglike , dtype=np.int),np.array(h_siglike, dtype=np.int)], index=["Background Like", "Signal Like"], columns=labels[:-1]+["Multi-nuclei"])


# In[12]:


total_evts = np.sum(h_siglike)+np.sum(h_bkglike)
h_bkglike_ratio_to_total = h_bkglike/total_evts
h_siglike_ratio_to_total = h_siglike/total_evts
table = pd.DataFrame([np.array(h_bkglike_ratio_to_total , dtype=np.float),np.array(h_siglike_ratio_to_total, dtype=np.float)], index=["Background Like", "Signal Like"], columns=labels[:-1]+["Multi-nuclei"])


# ## C and C efficiency ( divided by total events )

# In[13]:


ratio_C = [ h_siglike_ratio_to_total[index_C], h_bkglike_ratio_to_total[index_C]]
ratio_noC = [np.sum(h_siglike_ratio_to_total)-h_siglike_ratio_to_total[index_C], np.sum(h_bkglike_ratio_to_total)-h_bkglike_ratio_to_total[index_C]]
pd.DataFrame(np.array([ratio_C, ratio_noC]).T, columns=["C", "no-C"], index=["Signal Like", "Background Like"])


# ## C11 and no-C11 efficiency ( divided by total events)

# In[14]:


ratio_C11_bkglike = C11_bkglike/total_evts
ratio_noC11_bkglike = (np.sum(h_bkglike)-C11_bkglike)/total_evts
ratio_C11_siglike = C11_siglike/total_evts
ratio_noC11_siglike = (np.sum(h_siglike)-C11_siglike)/total_evts

pd.DataFrame(np.array([[ratio_C11_siglike, ratio_noC11_siglike], [ratio_C11_bkglike, ratio_noC11_bkglike]]), index=["Signal Like", "Background Like"], columns=["$^{11}C$", "no-$^{11}C$"])


# In[15]:


C_total = np.sum(h_C_bkglike)+np.sum(h_C_siglike)
print("C ratio:\t", C_total/total_evts)
C11_total = h_C_siglike[-1] + h_C_bkglike[-1]
no_C11_total = total_evts - C11_total
sub_ratio_C11_bkglike = C11_bkglike/C11_total
sub_ratio_noC11_bkglike = (np.sum(h_bkglike)-C11_bkglike)/no_C11_total
sub_ratio_C11_siglike = C11_siglike/C11_total
sub_ratio_noC11_siglike = (np.sum(h_siglike)-C11_siglike)/no_C11_total
print("C11_ratio:\t", C11_total/total_evts)
pd.DataFrame(np.array([[sub_ratio_C11_siglike, sub_ratio_noC11_siglike], [sub_ratio_C11_bkglike, sub_ratio_noC11_bkglike]]), index=["Signal Like", "Background Like"], columns=["$^{11}C$", "no-$^{11}C$"])

