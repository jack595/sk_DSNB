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
from IPython.display import display
import tqdm


def GetMinAndMax(df, key):
    if key in df.index:
        return ( df.loc[key]['min'], df.loc[key]['max'] )
    else:
        return ( 0, np.inf )

def GetVertexR(df):
    return np.sqrt(df["recX"]**2+df["recY"]**2+df["recZ"]**2)

def LoadPreviousPromptDelaySelectionResult(df_map, path_selection_result,
                                           index_to_select, name_selection):
    dir_select_result = LoadOneFileUproot(path_selection_result,
                               name_branch=f"{name_selection}Selection",
                               return_list=False)
    dir_select_result.pop("evtID")
    df_select_result = pd.DataFrame.from_dict(dir_select_result)
    df_map = pd.concat((df_map, df_select_result), axis=1)
    index_to_select = (index_to_select) & (df_map[f"Tag{name_selection}p"] == 0) &\
                      (df_map[f"Tag{name_selection}d"] == 0)
    return index_to_select



class PromptDelaySelection:
    def __init__(self, name_tag="IBD"):
        self.name_tag = f"{name_tag}Selection"
        self.name_tag_delay = f"Tag{name_tag}d"
        self.name_tag_prompt = f"Tag{name_tag}p"
        self.name_source_delay = f"{name_tag}Source"

        # Isolation cut
        self.name_tag_isolation = "TagSingle"

    def SetSelectionCriterion(self, path_xml:str):
        """

        :param path_xml: example:"/afs/ihep.ac.cn/users/j/junotemp006/junotemp006/myproject/SNSpecUnfold/channelClass/SNPSD/myJob/configFiles/IBD_select/scan_0_0.xml"
        :return:
        """
        df_parameters = pd.read_xml(path_xml,xpath="//condition").set_index("name")
        print("===>Selection Criterion:")
        df_parameters["max"] = df_parameters["max"].fillna(np.inf)
        df_parameters["min"] = df_parameters["min"].fillna(0)
        display(df_parameters)
        self.dR_cut = GetMinAndMax(df_parameters, "distance") # mm
        self.t_cut =  GetMinAndMax(df_parameters, "deltaT") # ns
        self.Ed_cut = GetMinAndMax(df_parameters, "denergy") # MeV
        self.Ed_cut2 = (5.2, 5.8) # MeV
        self.Ep_cut = GetMinAndMax(df_parameters, "penergy") # MeV
        self.R_FV_cut = GetMinAndMax(df_parameters, "VertexR")  # mm

    def DoSelection(self):

        v_IBDp_tag = np.ones(len(self.df_map_with_filter))*-1
        v_IBDd_tag = np.zeros(len(self.df_map_with_filter))
        v_IBDd_source = np.ones(len(self.df_map_with_filter))*-1

        for index, row in tqdm.tqdm( self.df_map_with_filter.iterrows() ):
            R_evt = GetVertexR(row)

            if (row["recE"]< self.Ep_cut[0]) \
                    or (row["recE"]> self.Ep_cut[1]) \
                    or (v_IBDd_tag[index]==1) \
                    or (R_evt >self.R_FV_cut[1]) \
                    or (R_evt <self.R_FV_cut[0]):
                v_IBDp_tag[index] = 0
                continue

            index_time = (self.df_map_with_filter["TriggerTime"]-row["TriggerTime"]>self.t_cut[0]) & (self.df_map_with_filter["TriggerTime"]-row["TriggerTime"]<self.t_cut[1])

            index_E_delay = ( (self.df_map_with_filter["recE"]<self.Ed_cut[1]) & (self.df_map_with_filter["recE"]>self.Ed_cut[0]) ) | \
                            ( (self.df_map_with_filter["recE"]<self.Ed_cut2[1]) & (self.df_map_with_filter["recE"]>self.Ed_cut2[0]) )

            dR = np.sqrt( (self.df_map_with_filter["recX"]-row["recX"])**2 + (self.df_map_with_filter["recY"]-row["recY"])**2 + (self.df_map_with_filter["recZ"]-row["recZ"])**2 )
            index_dR =  (dR < self.dR_cut[1]) & (dR>self.dR_cut[0])
            index_residual_delay_evt = (v_IBDd_tag!=1)

            index_delay_signal = (index_time & index_E_delay & index_dR & index_residual_delay_evt)

            if any(index_delay_signal):
                v_IBDp_tag[index] = 1
                v_IBDd_tag[np.where(index_delay_signal)[0][0]] = 1
                v_IBDd_source[np.where(index_delay_signal)[0][0]] = self.df_map_with_filter["evtID"][index]

            else:
                v_IBDp_tag[index] = 0


            # if index>2000:
            #     break

        self.df_map_with_filter[self.name_tag_delay] = np.array( v_IBDd_tag, dtype=np.int32 )
        self.df_map_with_filter[self.name_tag_prompt] = np.array( v_IBDp_tag, dtype=np.int32 )
        self.df_map_with_filter[self.name_source_delay] = np.array( v_IBDd_source, dtype=np.int32 )

        self.df_map_with_new_tag = pd.concat( (self.df_map, self.df_map_with_filter.set_index("evtID")[self.name_tag_prompt],
                                               self.df_map_with_filter.set_index("evtID")[self.name_tag_delay],
                                               self.df_map_with_filter.set_index("evtID")[self.name_source_delay]), axis=1 ).fillna(0)
        return self.df_map_with_new_tag

    def IsolationSelection(self):
        v_tag_singles = np.ones(len(self.df_map_with_filter))*-1

        for index, row in tqdm.tqdm(self.df_map_with_filter.iterrows()):
            R_evt = GetVertexR(row)

            if (row["recE"] < self.Ep_cut[0]) \
                    or (row["recE"] > self.Ep_cut[1]) \
                    or (R_evt > self.R_FV_cut[1]) \
                    or (R_evt < self.R_FV_cut[0]):
                v_tag_singles[index] = 0
                continue

            index_time = (self.df_map_with_filter["TriggerTime"] - row["TriggerTime"] > self.t_cut[0]) & (
                        self.df_map_with_filter["TriggerTime"] - row["TriggerTime"] < self.t_cut[1])

            dR = np.sqrt((self.df_map_with_filter["recX"] - row["recX"]) ** 2 + (
                            self.df_map_with_filter["recY"] - row["recY"]) ** 2 + (
                            self.df_map_with_filter["recZ"] - row["recZ"]) ** 2)

            index_dR = (dR < self.dR_cut[1]) & (dR > self.dR_cut[0])

            index_delay_signal = (index_time  & index_dR)

            v_tag_singles[index] = int(not np.any(index_delay_signal))
        self.df_map_with_filter[self.name_tag_isolation] = np.array(v_tag_singles, dtype=np.int32)

        display(self.df_map_with_filter.groupby(["TagSingle", "evtType"]).size())

        self.df_map_with_new_tag = pd.concat((self.df_map, self.df_map_with_filter.set_index("evtID")[self.name_tag_isolation] ),
                                    axis=1).fillna(0)
        return self.df_map_with_new_tag

    def SetDataset(self, df_evts:pd.DataFrame, index_filter:np.ndarray):
        self.df_map = df_evts
        self.df_map_with_filter = self.df_map[index_filter].sort_values(by=['TriggerTime']).reset_index()
        display(self.df_map_with_filter.groupby("evtType").size())
        # self.df_map_with_filter = self.df_map_with_filter.loc[:, ~self.df_map_with_filter.columns.duplicated()]

    def SaveTaggResults(self, path_save:str):
        # Save results into root file
        import ROOT
        from PandasTools import DataFrameToDict
        dir_save = DataFrameToDict( self.df_map_with_new_tag[["evtID",  self.name_tag_prompt,
                                                              self.name_tag_delay, self.name_source_delay]].astype(int) )
        rdf = ROOT.RDF.MakeNumpyDataFrame(dir_save)
        rdf.Snapshot(self.name_tag, path_save)
        rdf.Display().Print()

    def SaveTaggResultsSingles(self, path_save:str):
        # Save results into root file
        import ROOT
        from PandasTools import DataFrameToDict
        dir_save = DataFrameToDict( self.df_map_with_new_tag[["evtID",  self.name_tag_isolation]].astype(int) )
        rdf = ROOT.RDF.MakeNumpyDataFrame(dir_save)
        rdf.Snapshot(self.name_tag, path_save)
        rdf.Display().Print()

    def EvaluateTagging(self, name_truth_prompt="IBDp", name_truth_delay="IBDd"):
        # Add FV Cut
        v_R = GetVertexR(self.df_map_with_new_tag)
        index_FV_cut = (self.R_FV_cut[0]<v_R) & (v_R<self.R_FV_cut[1])

        df_map_IBDp_gf = self.df_map_with_new_tag.groupby([self.name_tag_prompt,"evtType"]).size()
        df_map_IBDd_gf = self.df_map_with_new_tag.groupby([self.name_tag_delay,"evtType"]).size()

        df_map_total = self.df_map_with_new_tag[index_FV_cut].groupby("evtType").size()


        print("IBD Prompt Signal Selection Efficiency:\t",  df_map_IBDp_gf.xs(1, level=self.name_tag_prompt).loc[name_truth_prompt]/np.sum(df_map_total.loc[name_truth_prompt]) )
        print("IBD Delay  Signal Selection Efficiency:\t",  df_map_IBDd_gf.xs(1, level=self.name_tag_delay).loc[name_truth_delay]/np.sum(df_map_total.loc[name_truth_delay]) )

        display( self.df_map_with_new_tag.groupby([self.name_tag_prompt,self.name_tag_delay,"evtType"]).size() )

    def EvaluateTaggingSingles(self):
        # Add FV Cut
        v_R = GetVertexR(self.df_map_with_new_tag)
        index_FV_cut = (self.R_FV_cut[0]<v_R) & (v_R<self.R_FV_cut[1])

        display( self.df_map_with_new_tag.groupby([self.name_tag_isolation,"evtType"]).size() )

        display( self.df_map_with_new_tag[index_FV_cut].groupby([self.name_tag_isolation,"evtType"]).size() )



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

    # For CC Selection
    parser.add_argument("--CCSelection", action="store_true", default=False, help="Do CC Selection After IBD Selection")
    parser.add_argument("--path-IBD", type=str,
                        default=None,
                        help="path of input about IBD Selection result")

    # For Isolation Selection
    parser.add_argument("--IsolationSelection", action="store_true", default=False,
                        help="Do Isolation selection to get eES and NC after CC Selection")
    parser.add_argument("--path-CC", type=str,
                        default=None,
                        help="path of input about CC Selection result")

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
    # Filter AfterPulse events with TagAP
    ## Default Configure: AfterPulse Selection before all selection
    index_to_select = (df_map["TagAP"]==0)


    if arg.CCSelection or arg.IsolationSelection:
        if arg.CCSelection:
            print("========================> Running CC Selection <======================")
        elif arg.IsolationSelection:
            print("========================> Running Isolation Selection <======================")
            # Load CC Selection Result
            if arg.path_CC == None:
                print(
                    "!!!!!ERROR:\n \tCC Selection needs inputting CC Selection results, use switch --path-CC to input the root file!")
                exit(0)
            index_to_select = LoadPreviousPromptDelaySelectionResult(df_map, arg.path_CC,
                                                    index_to_select, "CC")

        # Load IBD Selection Results
        if arg.path_IBD==None:
            print("!!!!!ERROR:\n \tSelection needs inputting IBD Selection result, use switch --path-IBD to input the root file!")
            exit(0)
        index_to_select = LoadPreviousPromptDelaySelectionResult(df_map, arg.path_IBD,
                                                    index_to_select, "IBD")

    # Filter pES events with PSD Tag
    if arg.AfterPSD or arg.CCSelection or arg.IsolationSelection:
        print("=============>  Do Selection After PSD  <=================")
        dir_PSD = LoadOneFileUproot(arg.path_PSDTools,
                                   name_branch="PSD", return_list=False)
        df_PSD = pd.DataFrame.from_dict(dir_PSD)
        df_PSD = df_PSD.rename({"evtType":"TagPSD"},axis=1)
        df_map = pd.concat( (df_map, df_PSD),axis=1)
        index_to_select = (index_to_select) & (df_map["TagPSD"]==1)

    # Apply SelectionTool
    if arg.IsolationSelection:
        name_Selection = 'Single'
        name_truth_prompt = ["C12", "eES"]
    elif arg.CCSelection:
        name_Selection = 'CC'
        name_truth_prompt = ["N12", "B12"]
        name_truth_delay = ["N12", "B12"]
    elif arg.IBDSelection:
        name_Selection = "IBD"
        name_truth_prompt = "IBDp"
        name_truth_delay = "IBDd"

    tool_selection = PromptDelaySelection(name_Selection)
    tool_selection.SetSelectionCriterion(arg.path_xml)
    tool_selection.SetDataset(df_evts=df_map, index_filter=index_to_select)
    if arg.IsolationSelection:
        tool_selection.IsolationSelection()
        tool_selection.SaveTaggResultsSingles(arg.outfile)
    else:
        tool_selection.DoSelection()
        tool_selection.SaveTaggResults(arg.outfile)

    if arg.IsolationSelection:
        tool_selection.EvaluateTaggingSingles()
    else:
        tool_selection.EvaluateTagging(name_truth_prompt=name_truth_prompt, name_truth_delay=name_truth_delay)



