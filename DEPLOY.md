# 使用说明

## 训练

conda activate base

export PYTHONPATH=./

python tools/train.py --config-file configs/xxx/xxx.py --num-gpus GpuNum

如：python tools/train.py --config-file configs/imotion/semseg-litept-small-v1m1-lidaronly-train-8gpu.py --num-gpus 8

注意：训练时，配置文件中的validation_mode可以设置为True，这样最终评测时就会去读取segLabel，并出评测报告。训练时候gpu数量发生变化时，需要调整配置文件中的总batch_size，目前测试下来是gpu数量的4倍比较合适，小于此值效果变差

## 推理

conda activate base

export PYTHONPATH=./

bash infer_zy_api.sh BagRoot BagName ResRoot GpuNum

如：bash infer_zy_api.sh /mlp/data_loop/workspace_wxp/test P\_SuZhou\_20250430-081111\_T22-7412 /mlp/data_loop/workspace_wxp/test/res 1

注意：单纯推理时，配置文件中的validation_mode需要设置为False，这样就不会去读取segLabel

## 验证/测试

训练：

conda activate base

export PYTHONPATH=./

如果在H20机器上训练，当前环境需要更改setup.py后重新编译pointrope和pointops

*cd libs/pointrope
rm -rf build/
find . -name ".so" -delete
find . -name "__pycache__" | xargs rm -rf
python setup.py install
cd ../..
cd libs/pointops
rm -rf build/
find . -name ".so" -delete
find . -name "__pycache__" | xargs rm -rf
python setup.py install
cd ../..***

python tools/train.py --config-file configs/xxx/xxx.py --num-gpus GpuNum --options save_path=xxx weight=xxx.pth

测试：

python tools/test.py --config-file configs/imotion/semseg-litept-small-v1m1-lidaronly.py --num-gpus 1 --options save_path=/mlp/data_loop/workspace_wxp/test-litept-infer/test-speed-bag/res/test-speed-bag weight=deploy/model_last-lidaronly.pth

注意：验证/测试时，配置文件中的validation_mode需要设置为True，这样最终评测时就会去读取segLabel，并出评测报告。另外需要将save_path下原有的内容清除(如有)，否则会导致沿用原有结果
