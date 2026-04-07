#! /bin/bash

data_dir=$1
dataset=$2
num_gpu=$3
data_name=${4:-"occ_frames"}
result_name=${5:-"occ_frames_denoise"}
model_dir="deploy/model_last-lidaronly.pth"
config_template="configs/imotion/semseg-litept-small-v1m1-lidaronly-denoise.py"

data_root=${data_dir}/${dataset}/${data_name}
result_dir=${data_dir}/${dataset}/${result_name}

# Check if dataset directory exists
if [ ! -d "${data_root}" ]; then
    echo "Path '${data_root}' does not exist!"
    exit 1
fi

# Model segmentation
if [ -d "${result_dir}" ]; then
    rm -rf ${result_dir}/*
fi


config_base="${config_template%.*}"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
config_file="${config_base}-${dataset}-${timestamp}.py"
cp "$config_template" "$config_file"
# Modify the data_root line in the new config file
sed -i "s#^data_root = .*#data_root = \"${data_root}\"#g" "${config_file}"

export PYTHONPATH=./
if ! python -u tools/test_denoise.py --config-file ${config_file} --num-gpus ${num_gpu} --options save_path=${result_dir} weight=${model_dir}; then
    echo "Error: occ-denoise infer failed!"
    exit 2
fi

# Concatenate pcd
python deploy/seg2pcd_denoise.py --result_base_dir ${result_dir}/result --data_base_dir ${data_root}

# delete infer-.npy,model results
rm -rf ${result_dir}/result
rm -rf ${result_dir}/model
rm -rf ${result_dir}/config.py
rm -rf ${result_dir}/test.log
