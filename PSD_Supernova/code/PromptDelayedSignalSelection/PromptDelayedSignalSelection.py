# -*- coding:utf-8 -*-
# @Time: 2022/4/8 14:11
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PromptDelayedSignalSelection.py
import matplotlib.pylab as plt
import numpy as np

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
import sys

sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/root_tool/python_script/")
import pandas as pd

def GetMinAndMax(df, key):
    return ( df.loc[key]['min'], df.loc[key]['max'] )

class PromptDelaySelection:
    def __init__(self, name_tag="IBD"):
        self.name_tag = f"{name_tag}Selection"
        self.name_tag_delay = f"Tag{name_tag}d"
        self.name_tag_prompt = f"Tag{name_tag}p"

    def SetSelectionCriterion(self, path_xml:str):
        """

        :param path_xml: example:"/afs/ihep.ac.cn/users/j/junotemp006/junotemp006/myproject/SNSpecUnfold/channelClass/SNPSD/myJob/configFiles/IBD_select/scan_0_0.xml"
        :return:
        """
        df_parameters = pd.read_xml(path_xml,xpath="//condition").set_index("name")
        self.dR_cut = GetMinAndMax(df_parameters, "distance") # mm
        self.t_cut =  GetMinAndMax(df_parameters, "deltaT") # ns
        self.Ed_cut = GetMinAndMax(df_parameters, "denergy") # MeV
        self.Ep_cut = GetMinAndMax(df_parameters, "penergy") # MeV

    def DoSelection(self):
        import tqdm

        v_IBDp_tag = np.ones(len(self.df_map_with_filter))*-1
        v_IBDd_tag = np.zeros(len(self.df_map_with_filter))

        for index, row in tqdm.tqdm( self.df_map_with_filter.iterrows() ):
            if row["recE"]< self.Ep_cut[0] or v_IBDd_tag[index]==1:
                v_IBDp_tag[index] = 0
                continue

            index_time = (self.df_map_with_filter["TriggerTime"]-row["TriggerTime"]>self.t_cut[0]) & (self.df_map_with_filter["TriggerTime"]-row["TriggerTime"]<self.t_cut[1])
            index_E_delay = ( (self.df_map_with_filter["recE"]<self.Ed_cut[1]) & (self.df_map_with_filter["recE"]>self.Ed_cut[0]) )
            dR = np.sqrt( (self.df_map_with_filter["recX"]-row["recX"])**2 + (self.df_map_with_filter["recY"]-row["recY"])**2 + (self.df_map_with_filter["recZ"]-row["recZ"])**2 )
            index_dR =  (dR < self.dR_cut[1]) & (dR>self.dR_cut[0])
            index_residual_delay_evt = (v_IBDd_tag!=1)

            index_delay_signal = (index_time & index_E_delay & index_dR & index_residual_delay_evt)

            if any(index_delay_signal):
                v_IBDp_tag[index] = 1
                v_IBDd_tag[np.where(index_delay_signal)[0][0]] = 1
            else:
                v_IBDp_tag[index] = 0

            # if index>2000:
            #     break

        self.df_map_with_filter[self.name_tag_delay] = np.array( v_IBDd_tag, dtype=np.int32 )
        self.df_map_with_filter[self.name_tag_prompt] = np.array( v_IBDp_tag, dtype=np.int32 )

        self.df_map_with_new_tag = pd.concat( (self.df_map, self.df_map_with_filter.set_index("evtID")[self.name_tag_prompt],
                                               self.df_map_with_filter.set_index("evtID")[self.name_tag_delay]), axis=1 ).fillna(0)
        return self.df_map_with_new_tag

    def SetDataset(self, df_evts:pd.DataFrame, index_filter:np.ndarray):
        self.df_map = df_evts
        self.df_map_with_filter = self.df_map[index_filter].reset_index()

    def SaveTaggResults(self, path_save:str):
        # Save results into root file
        import ROOT
        from PandasTools import DataFrameToDict
        dir_save = DataFrameToDict( self.df_map_with_new_tag[["evtID",  self.name_tag_prompt, self.name_tag_delay]].astype(int) )
        print(dir_save)
        rdf = ROOT.RDF.MakeNumpyDataFrame(dir_save)
        rdf.Snapshot(self.name_tag, path_save)
        rdf.Display().Print()

    def EvaluateTagging(self, name_truth_prompt="IBDp", name_truth_delay="IBDd"):
        from IPython.display import display
        df_map_IBDp_gf = self.df_map_with_new_tag.groupby([self.name_tag_prompt,"evtType"]).size()
        df_map_IBDd_gf = self.df_map_with_new_tag.groupby([self.name_tag_delay,"evtType"]).size()

        df_map_total = self.df_map_with_new_tag.groupby("evtType").size()

        print("IBD Prompt Signal Selection Efficiency:\t",  df_map_IBDp_gf.xs(1, level=self.name_tag_prompt).loc[name_truth_prompt]/df_map_total.loc[name_truth_prompt] )
        print("IBD Delay  Signal Selection Efficiency:\t",  df_map_IBDd_gf.xs(1, level=self.name_tag_delay).loc[name_truth_delay]/df_map_total.loc[name_truth_delay] )

        display( self.df_map_with_new_tag.groupby([self.name_tag_prompt,self.name_tag_delay,"evtType"]).size() )

if __name__ == '__main__':
    # Set Variables
    import argparse
    parser = argparse.ArgumentParser(description='ExtractFeatures to Separate AfterPulse')
    parser.add_argument("--path-PSDTools", type=str,
                        default="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/PSD/root/user_PSD_0_SN.root",
                        help="path of input about PSDTools")
    parser.add_argument("--path-evtTruth", type=str,
                        default="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/tag_event/root/sn_tag_0.root",
                        help="path of input about evtTruth")
    parser.add_argument("--path-AfterPulse",  type=str,
                    default="/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/AfterPulsePrediction/root/TagAfterPulse_0.root",
                    help="path of input about evtTruth")
    parser.add_argument("--outfile", "-o", type=str, default="try.root", help="name of outfile")
    parser.add_argument("--path-xml", type=str,
                        default="/afs/ihep.ac.cn/users/j/junotemp006/junotemp006/myproject/SNSpecUnfold/channelClass/SNPSD/myJob/configFiles/IBD_select/scan_0_0.xml",
                        help="XML file to set selection criteria")
    parser.add_argument("--AfterPSD", action="store_true", default=False, help="Prompt-Delayed Selection After PSD")
    arg = parser.parse_args()

    # Load Dataset
    from LoadMultiFiles import LoadOneFileUproot
    dir_map = LoadOneFileUproot(arg.path_evtTruth,
                                name_branch="evtTruth", return_list=False)
    dir_AP = LoadOneFileUproot(arg.path_AfterPulse,
                                name_branch="AfterPulseTag", return_list=False)

    df_AP = pd.DataFrame.from_dict(dir_AP)
    df_map = pd.DataFrame.from_dict(dir_map)
    df_map = pd.concat( (df_map, df_AP),axis=1)

    # Set Events to do the selection
    from collections import Counter
    index_to_select = (df_map["TagAP"]==0)
    if arg.AfterPSD:
        print("=============>  Do Selection After PSD  <=================")
        dir_PSD = LoadOneFileUproot(arg.path_PSDTools,
                                   name_branch="PSD", return_list=False)
        df_PSD = pd.DataFrame.from_dict(dir_PSD)
        df_PSD = df_PSD.rename({"evtType":"TagPSD"},axis=1)
        df_map = pd.concat( (df_map, df_PSD),axis=1)
        index_to_select = (index_to_select) & (df_map["TagPSD"]==1)


    # Apply SelectionTool
    tool_selection = PromptDelaySelection("IBD")
    tool_selection.SetSelectionCriterion(arg.path_xml)
    tool_selection.SetDataset(df_evts=df_map, index_filter=index_to_select)
    tool_selection.DoSelection()
    tool_selection.SaveTaggResults(arg.outfile)
    tool_selection.EvaluateTagging()



