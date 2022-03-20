#!/bin/bash
fileNo=$1
source /cvmfs/juno.ihep.ac.cn/sw/anaconda/Anaconda3-2020.11-Linux-x86_64/bin/activate root624 &&
python RunExtractFeactures.py -p "/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/PSD/root_noShift/user_PSD_{}__SN.root" \
-e "/afs/ihep.ac.cn/users/l/luoxj/PSD_Supernova/myJUNOCommon/share/tag_event/root/sn_tag_{}.root" -n $fileNo -o ./root/features_noShift_${fileNo}_{}.root &> ./log/log_${fileNo}.txt