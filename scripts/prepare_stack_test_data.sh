bag_name=P_GuangZhou_20260227-163322_AY7-0629_0
# repo_dir=/imotion/imotion/data-algo/workspace_wjg/data/occ_cases/stack/
debug_dir=/mlp/data_loop/workspace_wxp/test-litept-infer/occ_stack_debug_workspace/${bag_name}


rm /root/infer_results
infer_dir=$debug_dir/infer_results/
ln -s $infer_dir /root

rm /root/tmp
tmp_dir=$debug_dir/tmp
ln -s $tmp_dir /root
