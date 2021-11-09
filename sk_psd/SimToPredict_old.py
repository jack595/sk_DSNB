#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
parser = argparse.ArgumentParser(description='Sim to Prediction')
parser.add_argument('--start','-s', default=0, type=int, help='the start entry')
parser.add_argument('--end', '-e', default=2000, type=int, help="the end entry")
parser.add_argument('--iname', '-i', default=0, type=int, help="the name of outfile")
args = parser.parse_args()

import matplotlib.pylab as plt
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import numpy as np
def GetFileList(name_file_prefix:str, n_start, n_end):
    list_return = []
    for i in range(n_start, n_end+1):
        list_return.append(name_file_prefix+str(i).zfill(6)+".root")
    return list_return

# list_sim_data = GetFileList("root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/RawData_DSNB/AtmNC_v2/Model-G/atm_",
#                             n_start=8001, n_end=9000)
list_sim_data = GetFileList("root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/RawData_DSNB/AtmNC_v3/Model-G/atm_",
                            n_start=1, n_end=10)
print("files list:",list_sim_data)


# In[2]:


######### Load Data Tree ##########
import ROOT
chain_sim_data = ROOT.TChain("psdtree")
for i_name_file in list_sim_data:
    chain_sim_data.Add(i_name_file)
n_entries =  chain_sim_data.GetEntries()
print("entries:\t",n_entries)


# In[3]:


import sys
sys.path.append('/afs/ihep.ac.cn/users/l/luoxj/junofs_500G/sk_psd_DSNB/LS_ML/sk_psd/')
from GetPMTType import PMTType
pmt_type = PMTType()
from DSNB_dataset_sk import GetBins
(bins_hist, bins_hist_weightE) = GetBins()
import pickle
with open("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_time_jobs_DSNB_sk_data/model_maxtime_0.pkl",
          "rb") as fr:
    model_time = pickle.load(fr)


# In[18]:


dir_evts = {"equen":[], "edep":[],"pdg":[], "init_p":[], "vertex":[], "equen_trk":[], "edep_trk":[],
            "PSD":[], "hittime":[]}

import tqdm
#for i in range(n_entries):
for i in tqdm.trange(args.start, args.end):
    chain_sim_data.GetEntry(i)
    dir_evts["equen"].append(chain_sim_data.Eqen)
    # dir_evts["pdg_trk"].append(np.array(chain_sim_data.pdg_trk))
    dir_evts["pdg"].append(np.array(chain_sim_data.initpdg))
    dir_evts["vertex"].append(np.array([chain_sim_data.X, chain_sim_data.Y, chain_sim_data.Z]))
    dir_evts["init_p"].append(np.vstack((np.array(chain_sim_data.initpx), np.array(chain_sim_data.initpy),
                                                       np.array(chain_sim_data.initpz)))) #[i_evt][i_pdg][i_xyz]
    # dir_evts["equen_trk"].append(np.array(chain_sim_data.Qedep_trk))
    # dir_evts["edep_trk"].append(np.array(chain_sim_data.edep_trk))

    # Get PSD output
    v_flag_HAM = [pmt_type.GetPMTType(pmtid) for pmtid in chain_sim_data.PMTID ]
    v_flag_MCP = [not elem for elem in v_flag_HAM]
    hittime = np.array(chain_sim_data.Time)
    hist, bin_edeges = np.histogram(hittime, bins=bins_hist)
    predict_proba = model_time.predict_proba(np.array([hist/hist.max()]))
    dir_evts["PSD"].append(predict_proba[0][1])

    # Seperate two types of pmts
    # hist_HAM, bin_edges_HAM = np.histogram(hittime[v_flag_HAM], bins=bins_hist)
    # hist_MCP, bin_edges_MCP = np.histogram(hittime[v_flag_MCP], bins=bins_hist)
    # hist_HAM = hist_HAM/hist_HAM.max()
    # hist_MCP = hist_MCP/hist_MCP.max()

    # npes = chain_sim_data.Charge
    # hist_weightE_HAM, bin_edges_weightE_HAM = np.histogram(hittime[v_flag_HAM], bins=bins_hist_weightE, weights=npes[v_flag_HAM] )
    # hist_weightE_MCP, bin_edges_weightE_MCP = np.histogram(hittime[v_flag_MCP], bins=bins_hist_weightE, weights=npes[v_flag_MCP] )
    # dir_evts["hittime"].append(np.concatenate((hist_HAM, hist_MCP)))

# predict_proba = model_time.predict_proba(dir_evts["hittime"])



#print(dir_evts["equen_trk"])
#print(dir_evts["equen"])
print(dir_evts["PSD"])


# In[31]:


#dir_evts["equen"] = np.array(dir_evts["equen"])
#index_equen_cut = (dir_evts["equen"]<30) & (dir_evts["equen"]>10)
#plt.hist(np.array(dir_evts["PSD"])[index_equen_cut], bins=10)
#plt.semilogy()
np.savez("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/predict_v2/predict_"+str(args.iname)+".npz", dir_events=dir_evts)


# In[34]:


# f = np.load("./predict_v2/predict_0.npz", allow_pickle=True)
# print(f["dir_events"])

