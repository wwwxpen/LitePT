#! /bin/bash
bucket="imo-bev-lane-1322593371"
midDataBucket="imo-bev-mid-data-1322593371"
tmp_dir="tmp"
infer_dir="infer_results"

while getopts 's:f:d:b:g:p:' OPT; do
    case $OPT in
        s) stage="$OPTARG";;
        f) flow="$OPTARG";;
        d) dataset="$OPTARG";;
        b) AL_BB_file="$OPTARG";;
        g) GPU_NUM="$OPTARG";;
        p) parallel_cnt="$OPTARG";;
    esac
done
print_task_info() {
    echo "******** TASK INFO **********"
    echo "stage : ${stage}"
    echo "flow : ${flow}"
    echo "dataset : ${dataset}"
    echo "AL_BB_file : ${AL_BB_file}"
    echo "GPU_NUM : ${GPU_NUM}"
    echo "parallel_cnt : ${parallel_cnt}"
    echo "*****************************"
}
print_task_info



stage2() {
    # stacked frame denoise
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
    bash infer_api_denoise.sh ../${infer_dir}/OCC_stack_frame ${dataset} ${GPU_NUM}
    mv ../${infer_dir}/OCC_stack_frame/${dataset}/occ_frames ../${infer_dir}/OCC_stack_frame/${dataset}/occ_frames_denoise_bak
    mv ../${infer_dir}/OCC_stack_frame/${dataset}/occ_frames_denoise ../${infer_dir}/OCC_stack_frame/${dataset}/occ_frames

    # occlusion calculation
    TMP_FILE=../LitePT/"${dataset}.txt"
    touch ${TMP_FILE}
    find ../${infer_dir}/OCC_stack_frame/${dataset}/occ_frames -type f -name "*.pcd" | while read -r pcd_file; do
        echo "$pcd_file" >> "$TMP_FILE"
    done
    ./occ_SF_occlusion_api.sh ${TMP_FILE} ${GPU_NUM} true
    rm ${TMP_FILE}
    
}

if [ "$stage" == "2" ]; then
    stage2
fi