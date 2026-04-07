#!/bin/bash

downloadBucket="imo-bev-mid-data-1322593371"
cloudType="cloud"
tmp_dir="tmp"

# export LD_LIBRARY_PATH=/opt/conda/envs/slam/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=./

# Define the file path and number of lines per batch
dataPathListFile=$1
num_gpus=$2
is_zy=${3:-"false"}
N=$((30 * num_gpus))  # Number of lines to read each time

# Check if the file exists
if [[ ! -f "$dataPathListFile" ]]; then
    echo "File $dataPathListFile does not exist."
    exit 1
fi

# $dataset/data.json/data_uncorrected.json由宣哥这边拷贝到tmp/$dataset文件夹下
# # Read each line from the file
# while IFS= read -r line || [ -n "${line}" ]; do
#     dataset=$(echo "$line" | awk -F'/' '{print $(NF-2)}')
#     jsonFile="data_uncorrected.json"
#     if [ "$is_zy" == "true" ]; then
#         jsonFile="data.json"
#     fi
#     if [[ ! -d "../${tmp_dir}/${dataset}" || ! -f "../${tmp_dir}/${dataset}/${jsonFile}" ]]; then
#         python deploy/lidar_dataset.py --config_dir deploy/cloud.yaml --cloud_type ${cloudType} --bucket ${downloadBucket} --prefix ${dataset}/${jsonFile} --dir_path ../${tmp_dir}/${dataset}
#     fi
# done < "$dataPathListFile"

# Read the entire file into an array, one line per element
mapfile -t lines < "$dataPathListFile"

# Get the total number of lines in the file
totalLines=${#lines[@]}

# Calculate the number of batches
numBatches=$(( (totalLines + N - 1) / N ))

# Function to process each batch
process_batch() {
    local start=$1
    local end=$2
    for (( i=start; i<end && i<totalLines; i++ )); do
        # Process line ${lines[i]} here
        gpu_id=$((i % num_gpus))
        export CUDA_VISIBLE_DEVICES=$gpu_id
        ./occ_SF_occlusion_one_line.sh ${lines[i]} ${is_zy} &
        # Example processing command (replace with actual commands)
        # some_processing_command "${lines[i]}" &
    done

    # Wait for all background processes to complete
    wait
}

# Loop through each batch and process in parallel
for (( batch=0; batch<numBatches; batch++ )); do
    start=$((batch * N))
    end=$((start + N))
    if [[ $end -gt $totalLines ]]; then
        end=$totalLines
    fi
    # Start processing each batch in the background
    process_batch "$start" "$end"
done

echo "All batches processed."