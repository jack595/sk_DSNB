#!/bin/bash
fileNo=$1
options=$2
source /cvmfs/juno.ihep.ac.cn/sw/anaconda/Anaconda3-2020.11-Linux-x86_64/bin/activate root624 &&
python PromptDelayedSignalSelection.py --path-PSDTools /afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/PSD/root/user_PSD_${fileNo}_SN.root \
                                       --path-evtTruth /afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/tag_event/root/sn_tag_${fileNo}.root \
                                       --path-AfterPulse /afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/AfterPulsePrediction/root/TagAfterPulse_${fileNo}.root \
                                       --outfile ./try_${fileNo}.root \
                                       --path-xml /afs/ihep.ac.cn/users/j/junotemp006/junotemp006/myproject/SNSpecUnfold/channelClass/SNPSD/myJob/configFiles/IBD_select/scan_0_0.xml \
                                       $options

