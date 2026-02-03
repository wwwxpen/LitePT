_base_ = ["../_base_/default_runtime.py"]

# 【新增】 定义视觉模态的开关和参数
use_visual_modality = True  # 【总开关】：True=使用DINO图片特征，False=原始LitePT模式
dino_backbone_name = "dinov2_vitl14"
# 为不同大小的 dinov2 指定本地权重路径（按需修改为实际路径）
dinov2_local_weights = {
    "dinov2_vits14": "models/ditr/dinov2_vits14_pretrain.bin",  # small
    "dinov2_vitb14": "models/ditr/dinov2_vitb14_pretrain.bin",  # base
    "dinov2_vitl14": "models/ditr/dinov2_vitl14_pretrain.bin",  # large
}
dino_local_weight_path = dinov2_local_weights.get(dino_backbone_name, None)
dinov2_dims = {
    "dinov2_vits14": 384,   # small
    "dinov2_vitb14": 768,   # base
    "dinov2_vitl14": 1024,  # large
}
dino_dim = dinov2_dims.get(dino_backbone_name, 1024)  # ViT-L 的特征维度, 默认 1024
img_size = (378, 672)  # 输入图像尺寸, 论文中针对 nuScenes 的推荐尺寸,（按需调整以控制显存/性能）
vis_active=True  # 是否激活可视化
vis_output_dir="vis_ditr_output"  # 可选输出路径
vis_switches=dict(  # 从配置接收可视化开关
    save_raw_pcd=True,
    save_raw_img=True,
    save_proj=True,
    save_dino_map=True,
    save_dino_pcd=True,
    save_final_pcd=True
)

# misc custom setting
batch_size = 2  # bs: total bs in all gpus
num_worker = 4
mix_prob = 0.0 #0.8
empty_cache = False
enable_amp = True

save_path = "exp/nuscenes/semseg-litept-small-v1m1-lessworker-fusion"

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=16,
    backbone_out_channels=72,
    backbone=dict(
        # 【修改】 将类型改为我们将要实现的 PDITR 类名，注意：如果开关关闭，虽然用的是新类，但逻辑应回退到原始 LitePT
        type="PDITR_LitePT",
        # 【新增】 传入视觉参数
        use_visual_modality=use_visual_modality,
        dino_backbone_name=dino_backbone_name,
        dino_dim=dino_dim,
        img_size=img_size,
        # 从配置文件传入 local weight path
        dino_local_weight_path=dino_local_weight_path,

        vis_active=vis_active,
        vis_output_dir=vis_output_dir,
        vis_switches=vis_switches,

        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "NuScenesDataset"
data_root = "data/nuscenes"
ignore_index = -1
names = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

# 【新增】 动态定义 Collect 需要收集的 Key，如果开启视觉，需要多收集 imgs, intrinsics, extrinsics
collect_keys = ("coord", "color", "grid_coord", "segment")
if use_visual_modality:
    collect_keys = collect_keys + ("imgs", "intrinsics", "extrinsics")

data = dict(
    num_classes=16,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,

        # 【新增】 传入参数告诉 Dataset 是否读图
        load_camera=use_visual_modality, 
        img_size=img_size,

        transform=[
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": 0.05}),
            dict(
                type="Collect",

                # 【修改】 使用上面动态定义的 keys
                keys=collect_keys,

                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,

        # 【新增】 验证集也需要读图
        load_camera=use_visual_modality,
        img_size=img_size,

        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="ToTensor"),
            dict(
                type="Collect",

                # 【修改】 增加视觉相关的 key，保留 inverse 等验证需要的 key
                keys=collect_keys + ("origin_segment", "inverse"),

                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,

        # 【新增】 测试集
        load_camera=use_visual_modality,
        img_size=img_size,

        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",

                    # 【修改】 增加视觉相关的 key
                    keys = collect_keys + ("index",),

                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=3),
    dict(type="PreciseEvaluator", test_last=False),
]