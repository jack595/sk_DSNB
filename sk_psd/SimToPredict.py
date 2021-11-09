#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
parser = argparse.ArgumentParser(description='Sim to Prediction')
parser.add_argument('--start','-s', default=0, type=int, help='the start entry')
parser.add_argument('--end', '-e', default=2000, type=int, help="the end entry")
parser.add_argument('--iname', '-i', default=0, type=int, help="the name of outfile")
parser.add_argument("--particle", "-p", type=str, default="Neutron", help="which type particle to deal with") # AmBe, Neutron
args = parser.parse_args()

import matplotlib.pylab as plt
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

import numpy as np

def GetGammaRatio(v_pdg, v_E_pdg, v_Equen):
    index_lepton = (v_pdg==22) | (v_pdg==11) | (v_pdg==-11)
    index_hadron = [not item for item in index_lepton]
    v_ratio = []
    for i in range(len(v_E_pdg)):
        # v_ratio.append(np.sum(self.evts_to_fit["equen_pdgdep"][i][index_lepton])/np.sum(self.evts_to_fit["equen_pdgdep"][i]) )
        v_ratio.append(
        1-np.sum(v_E_pdg[i][index_hadron]) / v_Equen[i])
    v_ratio = np.array(v_ratio)
    return v_ratio

def GetFileList(name_file_prefix:str, n_start, n_end, particle_type="atmNC"):
    list_return = []
    if particle_type == "atmNC":
        for i in range(n_start, n_end+1):
            list_return.append(name_file_prefix+str(i).zfill(6)+".root")
        return list_return
    else:
        for i in range(n_start, n_end+1):
            list_return.append(name_file_prefix+str(i)+".root")
        return list_return

# list_sim_data = GetFileList("root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/RawData_DSNB/AtmNC_v2/Model-G/atm_",
#                             n_start=8001, n_end=9000)
# list_sim_data = GetFileList("root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/RawData_DSNB/AtmNC_v3/Model-G/atm_",
#                             n_start=1, n_end=2000)



if args.particle == "Neutron":
    list_sim_data = GetFileList("root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/RawData_DSNB/Neutron/neutron_",
                            n_start=1, n_end=10000, particle_type="Neutron")
    # list_sim_data = GetFileList("root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/RawData_DSNB/Neutron/neutron_",
    #                             n_start=1, n_end=10, particle_type="Neutron")
elif args.particle == "AmBe_0_0_0":
    list_sim_data = GetFileList(f"root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/RawData_DSNB/{args.particle}/{args.particle}_",
                                n_start=1, n_end=2000, particle_type=args.particle)
else:
    # list_sim_data = GetFileList(f"root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/RawData_DSNB/{args.particle}/{args.particle.split('_')[0]}_",
    #                             n_start=1, n_end=1100, particle_type=args.particle)
    list_sim_data = GetFileList(f"root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/RawData_DSNB/{args.particle}/{args.particle.split('_')[0]}_",
                                n_start=1, n_end=2000, particle_type=args.particle)

# list_sim_data = GetFileList("root://junoeos01.ihep.ac.cn//eos/juno/user/luoxj/RawData_DSNB/AtmNC/Model-G/atm_",
#                             n_start=1, n_end=100)
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

with open("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_WeightEtime_jobs_DSNB_sk_data/model_maxtime_0.pkl",
          "rb") as fr:
    model_time_with_charge = pickle.load(fr)

with open("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/model_maxtime_combine_jobs_DSNB_sk_data/model_maxtime_0.pkl",
          "rb") as fr:
    model_time_combine= pickle.load(fr)

# In[4]:


dir_evts = {"equen":[], "edep":[],"pdg":[], "init_p":[], "vertex":[], "pdg_pdgdep":[],"equen_pdgdep":[], "edep_pdgdep":[],
            "PSD":[], "h_time":[],"PSD_with_charge":[], "h_time_with_charge":[],"name_file_source":[], "entry_source":[],
            "h_time_truth":[], "h_time_with_charge_truth":[], "lepton_ratio":[],"PSD_combine":[]}

load_truth = True

import tqdm
#for i in range(n_entries):
for i in tqdm.trange(args.start, args.end):
    chain_sim_data.GetEntry(i)
    # try:
    if True:
        if load_truth:
            hittime_truth = np.array(chain_sim_data.Time_Truth)
            charge_truth = np.array(chain_sim_data.Charge_Truth)
        
        hittime = np.array(chain_sim_data.Time)
        charge = np.array(chain_sim_data.Charge)

        dir_evts["equen_pdgdep"].append(np.array(chain_sim_data.Qedep_pdgdep))
        dir_evts["edep_pdgdep"].append(np.array(chain_sim_data.edep_pdgdep))
        dir_evts["pdg_pdgdep"].append(np.array(chain_sim_data.pdg_pdgdep))
        dir_evts["equen"].append(chain_sim_data.Eqen)
        # dir_evts["pdg_pdgdep"].append(np.array(chain_sim_data.pdg_pdgdep))
        dir_evts["pdg"].append(np.array(chain_sim_data.initpdg))
        dir_evts["vertex"].append(np.array([chain_sim_data.X, chain_sim_data.Y, chain_sim_data.Z]))
        dir_evts["init_p"].append(np.vstack((np.array(chain_sim_data.initpx), np.array(chain_sim_data.initpy),
                                                           np.array(chain_sim_data.initpz)))) #[i_evt][i_pdg][i_xyz]

        try:
        # if True:
            dir_evts["name_file_source"].append(str(chain_sim_data.name_file_source))
            dir_evts["entry_source"].append(int(chain_sim_data.entry_source))
        except :
            pass


        # Get Track Information


        # Get PSD output
        # v_flag_HAM = [pmt_type.GetPMTType(pmtid) for pmtid in chain_sim_data.PMTID ]
        # v_flag_MCP = [not elem for elem in v_flag_HAM]

        hist, bin_edeges = np.histogram(hittime, bins=bins_hist)
        hist_with_charge, bin_edeges_with_charge = np.histogram(hittime, bins=bins_hist, weights=charge)

        # h_time PSD
        predict_proba = model_time.predict_proba(np.array([hist/hist.max()]))
        dir_evts["PSD"].append(predict_proba[0][1])
        dir_evts["h_time"].append(hist)

        # h_time_with_charge PSD
        predict_proba_with_charge = model_time_with_charge.predict_proba(np.array([hist_with_charge/hist_with_charge.max()]))
        dir_evts["PSD_with_charge"].append(predict_proba_with_charge[0][1])
        dir_evts["h_time_with_charge"].append(hist_with_charge)

        # h_time_combine PSD
        h_combine = np.concatenate((hist/hist.max(),hist_with_charge/hist_with_charge.max()))
        predict_proba_combie = model_time_combine.predict_proba(np.array([h_combine]))
        dir_evts["PSD_combine"].append(predict_proba_combie[0][1])


        if load_truth:
            hist_truth, bin_edeges_truth = np.histogram(hittime_truth, bins=bins_hist)
            hist_with_charge_truth, bin_edeges_with_charge_truth = np.histogram(hittime_truth, bins=bins_hist, weights=charge_truth)
        

        # h_time truth
        dir_evts["h_time_truth"].append(hist_truth)
        dir_evts["h_time_with_charge_truth"].append(hist_with_charge_truth)
        

        # Seperate two types of pmts
        # hist_HAM, bin_edges_HAM = np.histogram(hittime[v_flag_HAM], bins=bins_hist)
        # hist_MCP, bin_edges_MCP = np.histogram(hittime[v_flag_MCP], bins=bins_hist)
        # hist_HAM = hist_HAM/hist_HAM.max()
        # hist_MCP = hist_MCP/hist_MCP.max()

        # npes = chain_sim_data.Charge
        # hist_weightE_HAM, bin_edges_weightE_HAM = np.histogram(hittime[v_flag_HAM], bins=bins_hist_weightE, weights=npes[v_flag_HAM] )
        # hist_weightE_MCP, bin_edges_weightE_MCP = np.histogram(hittime[v_flag_MCP], bins=bins_hist_weightE, weights=npes[v_flag_MCP] )
        # dir_evts["hittime"].append(np.concatenate((hist_HAM, hist_MCP)))
    # except Exception:
    #     print(f"Something Go Wrong in entry {i}, Continue!!!")
    #     continue

dir_evts["lepton_ratio"] = GetGammaRatio(np.array(dir_evts["pdg_pdgdep"][0]), np.array(dir_evts["equen_pdgdep"]), np.array(dir_evts["equen"]))
if dir_evts["name_file_source"] == []:
    del dir_evts["name_file_source"]
    del dir_evts["entry_source"]
# plt.hist(dir_evts["lepton_ratio"])
# plt.show()


# print(dir_evts["equen_pdgdep"])
# print(dir_evts["equen"])
# print(dir_evts["PSD"])
# print(dir_evts["PSD_with_charge"])
# plt.hist(dir_evts["PSD_with_charge"], bins=20)
# plt.show()

# In[ ]:


# dir_evts["equen"] = np.array(dir_evts["equen"])
# index_equen_cut = (dir_evts["equen"]<30) & (dir_evts["equen"]>10)
# plt.hist(np.array(dir_evts["PSD"])[index_equen_cut], bins=10)
# plt.semilogy()
if args.particle == "Neutron":
    np.savez("/afs/ihep.ac.cn/users/l/luoxj/sk_psd/predict_withpdgdep/predict_"+str(args.iname)+".npz", dir_events=dir_evts)
else:
    np.savez(f"/afs/ihep.ac.cn/users/l/luoxj/sk_psd/predict_withpdgdep_{args.particle}/predict_"+str(args.iname)+".npz", dir_events=dir_evts)

# In[ ]:


# f = np.load("./predict_v2/predict_0.npz", allow_pickle=True)
# print(f["dir_events"])

