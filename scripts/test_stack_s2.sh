dataset_name=P_GuangZhou_20260227-163322_AY7-0629_0
od_txt_path=/mlp/data_loop/workspace_wxp/test-litept-infer/occ_stack_debug_workspace/${dataset_name}/dps_od_bbox_result.txt
gpu_num=2
parallel_cnt=32
# /root/Pointcept/occ_stack_s2.sh -f imotion -d ${dataset_name} -b ${od_txt_path} -g ${gpu_num} -p ${parallel_cnt} -s 2
/root/LitePT/occ_stack_s2.sh -f imotion -d ${dataset_name} -b ${od_txt_path} -g ${gpu_num} -p ${parallel_cnt} -s 2
