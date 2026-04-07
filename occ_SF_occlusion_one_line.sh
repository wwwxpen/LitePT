#! /bin/bash

downloadBucket="imo-bev-mid-data-1322593371"
cloudType="cloud"
tmp_dir="tmp"
line=$1
is_zy=${2:-"false"}

dataset=$(echo "$line" | awk -F'/' '{print $(NF-2)}')
second_last_slash=${line%/*}
baseFolder=${second_last_slash%/*}
savePath="${baseFolder}/occ_occlusion"
plate_num="others"
if [[ $dataset == *"_6083_"* ]]; then
    plate_num="6083"
fi
if [[ $dataset == *"_0909_"* ]]; then
    plate_num="0909"
fi

if [ "$is_zy" == "true" ]; then
    python deploy/cuda_occlusion.py --frame_pcd_dir $line --data_json_dir ../${tmp_dir}/${dataset}/data.json --save_path ${savePath} --car_plate_num ${plate_num} --voxelSize 0.05 --is_zy
else
    python deploy/cuda_occlusion.py --frame_pcd_dir $line --data_json_dir ../${tmp_dir}/${dataset}/data_uncorrected.json --save_path ${savePath} --car_plate_num ${plate_num} --voxelSize 0.05
fi

