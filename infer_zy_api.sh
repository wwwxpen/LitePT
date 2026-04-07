#! /bin/bash

data_dir=$1
dataset=$2
result_dir=$3
model_dir="deploy/model_last-lidaronly.pth"
config_template="configs/imotion/semseg-litept-small-v1m1-lidaronly.py"
# config_template="configs/imotion/semseg-litept-small-v1m1-fusion.py"
num_gpu=$4

data_root=${data_dir}/${dataset}/samples

# Check if dataset directory exists
if [ ! -d "${data_root}" ]; then
    echo "Path '${data_root}' does not exist!"
    exit 1
else
    items=$(ls ${data_root}/)
    count=$(echo "$items" | wc -l)
    if [ $count -eq 0 ]; then
        echo "${dataset} has no scenes"
        exit 2
    else
        for scene in `ls ${data_root}/`; do
            if [ ! -d "${data_root}/${scene}/lidarFusion_pcd" ]; then
                echo "Path '${data_root}/${scene}/lidarFusion_pcd' does not exist!"
                exit 3
            fi
        done
    fi
fi

# Model segmentation
if [ -d "${result_dir}/${dataset}/result" ]; then
    rm -rf ${result_dir}/${dataset}/result
fi
if [ -d "${result_dir}/${dataset}/seged_pcds" ]; then
    rm -rf ${result_dir}/${dataset}/seged_pcds
fi

config_base="${config_template%.*}"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
config_file="${config_base}-${dataset}-${timestamp}.py"
cp "$config_template" "$config_file"
# Modify the data_root line in the new config file
sed -i "s#^data_root = .*#data_root = \"${data_root}\"#g" "${config_file}"

export PYTHONPATH=./

# check if has toplidar
python -u check_lidarTop.py --data_path ${data_dir}/${dataset}
hasToplidar=$?
if [ "$hasToplidar" -eq 1 ]; then echo "点云包含Toplidar"; else echo "点云不包含Toplidar"; fi

# if not use AT128 lidar, then re-generate pcd in LidarFusion_pcd file, if no Toplidar, then only use AT128 lidar
if [ "$hasToplidar" -eq 1 ]; then
    python check_if_use_AT128.py  --data_path ${data_dir}/${dataset}
fi

# infer
if ! python -u tools/test_zy.py --config-file ${config_file} --num-gpus ${num_gpu} --options save_path=${result_dir}/${dataset} weight=${model_dir}; then
    echo "Error: occ-single infer failed!"
    exit 5
fi

# Concatenate pcd
python deploy/seg2pcd.py --result_base_dir ${result_dir}/${dataset}/result --data_base_dir ${data_root}

# Check if segmentation valid, and generate bad pcd list file
seg_folder_path=${result_dir}/${dataset}/seged_pcds
bad_pcd_file_path=${result_dir}/${dataset}/bad_pcd.txt
python check_AL_valid_api.py --seg_folder_path ${seg_folder_path} --output_file_path ${bad_pcd_file_path}

# Reinfer bad pcd, just one cycle now
max_reinfer_cycle=1
reinfer_path=${result_dir}/${dataset}/reinfer
data_root_reinfer=${result_dir}/${dataset}/bad_pcd_to_reinfer
for ((i=1; i<=max_reinfer_cycle; i++)); do
    if python copy_bad_pcd_to_new_root.py --data_root ${data_root} --bad_pcd_txt_path ${bad_pcd_file_path} --data_root_reinfer ${data_root_reinfer}; then
        rm -fr ${bad_pcd_file_path}
        break # no pcd to reinfer
    else
        echo "reinfer_cycle $i start"
        config_file_reinfer="${config_base}-${dataset}-${timestamp}-reinfer.py"
        cp "$config_template" "$config_file_reinfer"
        sed -i "s#^data_root = .*#data_root = \"${data_root_reinfer}\"#g" "${config_file_reinfer}"
        # reinfer
        python tools/test_zy.py --config-file ${config_file_reinfer} --num-gpus ${num_gpu} --options save_path=${reinfer_path} weight=${model_dir}        
        # seg pcd and check if segmentation valid
        python deploy/seg2pcd.py --result_base_dir ${reinfer_path}/result --data_base_dir ${data_root_reinfer}
        python check_AL_valid_api.py --seg_folder_path ${reinfer_path}/seged_pcds --output_file_path ${bad_pcd_file_path} --no_delete_pcd
        # copy new infered seged pcd to origin path
        for file in ${reinfer_path}/seged_pcds/*; do
            mv "$file" ${seg_folder_path}
            echo "move reinfer seged pcd: $(basename "$file") to ${seg_folder_path}"
        done
    fi
    rm -fr ${reinfer_path} ${data_root_reinfer} ${bad_pcd_file_path}
done

# filter corner radar point
if [ "$hasToplidar" -eq 1 ]; then
    python remove_some_cornerlidar_point.py --data_path ${data_dir}/${dataset} --seg_folder_path ${seg_folder_path}
fi