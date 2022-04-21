#!/bin/bash
fileNo=$1
options=$2
export PYTHONPATH=/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/code/PromptDelayedSignalSelection/pyinstalled:$PYTHONPATH
source /cvmfs/juno.ihep.ac.cn/sw/anaconda/Anaconda3-2020.11-Linux-x86_64/bin/activate root624 &&
python PromptDelayedSignalSelection.py --path-PSDTools /afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/PSD/root/user_PSD_${fileNo}_SN.root \
                                       --path-evtTruth /afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/tag_event/root/sn_tag_${fileNo}.root \
                                       --path-AfterPulse /afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/AfterPulsePrediction/root/TagAfterPulse_${fileNo}.root \
                                       --outfile ./try_${fileNo}.root \
                                       --path-xml /afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/code/PromptDelayedSignalSelection/CCSelectionCriteria.xml \
                                       --path-IBD /afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/code/PromptDelayedSignalSelection/root_IBDSelection/try_1.root \
                                       --CCSelection\
                                       $options
