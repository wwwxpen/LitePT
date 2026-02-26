import torch
import torch.nn as nn
from functools import partial
import random
import numpy as np

from models.builder import MODELS
# 导入 PTv3
from ..litept.litept import LitePT
from models.utils import offset2batch
from models.utils.structure import Point
from .ditr_utils import DINOFeatureExtractor, DITRInjector
from .ditr_vis import DITRVisualizer

from torch_scatter import scatter_max

@MODELS.register_module("PDITR_LitePT")
class PDITR_LitePT(LitePT):
    def __init__(self, 
                 use_visual_modality=True, 
                 dino_backbone_name="dinov2_vitl14", 
                 dino_local_weight_path=None, # 本地权重路径
                 dino_dim=1024,
                 img_size=(378, 672),
                 vis_switches=None,         # 从配置接收可视化开关
                 vis_active=True,           # 是否激活可视化
                 vis_output_dir="vis_ditr_output",  # 可选输出路径
                 **kwargs):
        super().__init__(**kwargs)
        self.use_visual_modality = use_visual_modality
        self.img_size = img_size
        
        if self.use_visual_modality:
            # 注意：这里我们使用的是 ditr_utils 里修改过(支持本地权重)的 DINOFeatureExtractor
            print(f"[PDITR] Initializing DINOv2: {dino_backbone_name}")
            self.dino_extractor = DINOFeatureExtractor(dino_backbone_name, dino_local_weight_path)
            self.injector = DITRInjector(self.dino_extractor, debug=False)

            self.patch_size = 14 # DINOv2 默认为 14
            
            # 如果配置里没有提供 vis_switches，就使用默认值
            default_vis_switches = {
                "save_raw_pcd": False, 
                "save_raw_img": False,
                "save_proj": False,
                "save_dino_map": False,
                "save_dino_pcd": False,
                "save_final_pcd": False
            }
            vis_cfg = vis_switches if vis_switches is not None else default_vis_switches
            # active=True 开启保存，active=False 关闭所有保存
            if vis_active:
                self.vis = DITRVisualizer(output_dir=vis_output_dir, active=vis_active, switches=vis_cfg)
            else:
                self.vis = None

            self.dino_projections = nn.ModuleList()
            
            # 动态获取每一层 Decoder 需要的维度
            for i in range(len(self.dec)):
                # self.dec[i] 是一个 Stage
                # self.dec[i][0] 是 Unpooling 模块
                up_layer = self.dec[i][0] 
                
                # Unpooling 里的 proj 是一个 PointSequential，里面包含 Linear, Norm, Act 等
                # 我们通过遍历 modules() 找到第一个 nn.Linear 层
                out_dim = None
                for m in up_layer.proj.modules():
                    if isinstance(m, nn.Linear):
                        out_dim = m.out_features
                        break
                
                if out_dim is None:
                    # 如果万一没找到，打印错误信息帮助调试
                    raise AttributeError(f"Could not find nn.Linear layer in decoder stage {i} projection: {up_layer.proj}")
                
                self.dino_projections.append(
                    nn.Sequential(
                        nn.Linear(dino_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.GELU()
                    )
                )

    def forward(self, data_dict):
        # 预处理：如果是测试模式且关闭了视觉，直接走 LitePT 原生路径
        if not self.use_visual_modality:
            return super().forward(data_dict)
        #  可视化
        if self.use_visual_modality and hasattr(self, 'vis') and self.vis is not None:
            # 使用batch_index或scene_id作为frame_id
            frame_id = None
            if "batch_index" in data_dict:
                frame_id = data_dict["batch_index"][0].item()
            elif "scene_id" in data_dict:
                frame_id = data_dict["scene_id"][0].item()
            self.vis.start_new_frame(frame_id=frame_id)

        # ... (forward 函数保持之前的 Max Pooling 版本不变) ...
        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        
        # 1. 初始 DINO 特征提取
        dino_feat_current = None
        dino_feat_pyramid = []
        
        if self.use_visual_modality and "imgs" in data_dict:
            imgs = data_dict["imgs"]
            # print(f"Input images shape: {imgs.shape}")  # [B, V, C, H, W]
            img_feats_maps = self.injector.dino(imgs)
            # print(f"DINO feature maps shape: {img_feats_maps.shape}")  # [B, V, Dim, PH, PW]

            # 【修改】传入 segment (labels) 给 sample_from_maps
            # 注意：Point 类封装后，point.segment 或者 data_dict['segment'] 都可以
            # 但这里 point 已经被 sparsify 了，为了画原始点云，最好用 data_dict['segment'] 
            # 不过 data_dict['coord'] 和 point.coord 在 sparsify 后可能不一样
            # 为了对齐，我们使用 point.segment (如果存在)
            labels = point.segment if "segment" in point.keys() else None

            # 【关键逻辑】使用 color 字段进行投影
            # 1. 为什么用 color？因为它在 Dataset 阶段被赋值为增强前的原始 coord。
            # 2. 为什么不直接用 coord？因为 coord 经历了数据增强（平移旋转），与相机外参不再对齐。
            # 3. 为什么不自定义字段？因为框架只对 color 等标准字段在 Sparsify 时进行同步下采样。
            proj_coord = point.color if "color" in point.keys() else point.coord

            dino_feat_current = self.sample_from_maps(
                proj_coord, 
                offset2batch(point.offset), 
                img_feats_maps, 
                data_dict["cam_params"], 
                data_dict["extrinsics"],
                data_dict["img_target_size"], # [B, 2]
                imgs,   # 【新增】传入原始图片用于可视化
                labels  # 【新增】传入 Label 用于可视化
            )
            dino_feat_pyramid.append(dino_feat_current)

        # 2. Encoder (同步 Max Pooling)
        # 必须手动遍历 Encoder 以插入 Pooling 逻辑
        # 注意：这里需要根据 v3m1_base 的结构，不要调用 self.enc(point)
        # 而是遍历 self.enc
        
        # Point Embedding
        point = self.embedding(point)
        
        # 遍历 Encoder Stages
        for s, enc_stage in enumerate(self.enc):
            if s > 0:
                # 获取 Pooling 层
                down_layer = enc_stage[0] 
                point = down_layer(point)
                
                # DINO Max Pooling
                if dino_feat_current is not None:
                    if hasattr(point, "pooling_inverse"):
                        cluster_idx = point.pooling_inverse
                        dino_feat_next, _ = scatter_max(
                            dino_feat_current, 
                            cluster_idx, 
                            dim=0
                        )
                        dino_feat_current = dino_feat_next
                        dino_feat_pyramid.append(dino_feat_current)
                
                # 运行该 Stage 剩余的 Block
                for i, block in enumerate(enc_stage):
                    if i == 0: continue # 跳过第一个 (down_layer)
                    point = block(point)
            else:
                # s=0 没有 downsample
                point = enc_stage(point)
                # dino_feat_pyramid[0] 已在最开始添加

        # 3. Decoder
        if not self.enc_mode:
            for i, dec_stage in enumerate(self.dec):
                unpool_layer = dec_stage[0]
                
                if self.use_visual_modality and len(dino_feat_pyramid) > 0:
                    # 计算对应的 Pyramid 层级
                    # i=0 (deepest) -> target=L3 (if total 5 levels)
                    # 倒数第 (i+2) 个特征
                    target_idx = -(i + 2)
                    
                    if abs(target_idx) <= len(dino_feat_pyramid):
                        dino_feat_target = dino_feat_pyramid[target_idx]
                        
                        # 投影
                        dino_feat_proj = self.dino_projections[i](dino_feat_target)
                        
                        # 手动 Unpooling
                        parent = point.pop("pooling_parent")
                        inverse = point.pooling_inverse
                        
                        # 特征融合: Up + Skip + DINO
                        parent = unpool_layer.proj_skip(parent) # Skip
                        parent.feat = parent.feat + dino_feat_proj # Inject DINO
                        
                        point_feat_up = unpool_layer.proj(point).feat # Up
                        parent.feat = parent.feat + point_feat_up[inverse]
                        
                        if unpool_layer.traceable:
                            parent["unpooling_parent"] = point
                        
                        point = parent
                    else:
                        # 容错
                        point = unpool_layer(point)
                else:
                    point = unpool_layer(point)
                
                for j, block in enumerate(dec_stage):
                    if j == 0: continue # 跳过第一个 (unpool_layer)
                    point = block(point)

        # 可视化
        if self.use_visual_modality and hasattr(self, 'vis') and self.vis is not None:
            self.vis.reset_for_next_frame()

        return point


    def project_points_unified(self, pc_cam, params, target_size):
        """
        通用投影函数 (PyTorch 实现)
        pc_cam: [N, 3] 相机坐标系下的点
        params: 单个相机的参数字典
        target_size: [H, W] 目标图像尺寸
        """
        model = params['model']
        H_target, W_target = target_size
        H_base, W_base = params['base_size']
        scale_h = H_target / H_base
        scale_w = W_target / W_base

        if model == "mei":
            # 对应 mei_pro.py 逻辑
            ksi = params['ksi']
            k = params['k']
            if len(k) == 4:
                k.append(0.0)
            k1, k2, p1, p2, k3 = k
            u0, v0 = params['center']
            gama1, gama2 = params['gama']
            gama1 = gama1 * gama2 # 按照 mei_pro.py: gama1 = gama1 * gama2

            # 归一化到单位球面
            dist = torch.norm(pc_cam, dim=1, keepdim=True)
            pts_norm = pc_cam / (dist + 1e-6)
            
            # 投影到平面
            x_mu = pts_norm[:, 0:1] / (pts_norm[:, 2:3] + ksi)
            y_mu = pts_norm[:, 1:2] / (pts_norm[:, 2:3] + ksi)
            
            # 畸变矫正 (按照 mei_pro.py 的 Xmd 逻辑)
            rho2 = x_mu**2 + y_mu**2
            # temp = 1 + k1*r^2 + k2*r^4 + k3*rho^3
            temp = 1 + k1*rho2 + k2*(rho2**2) + k3*(rho2**3)
            
            # Xmd = Xmu * temp + 2*k3*Xmu*Ymu + k4*(rho + 2*Xmu^2)
            u_distort = x_mu * temp + 2*p1*x_mu*y_mu + p2*(rho2 + 2*x_mu**2)
            v_distort = y_mu * temp + 2*p2*x_mu*y_mu + p1*(rho2 + 2*y_mu**2)
            
            # 映射到像素并缩放
            u = (u_distort * gama1 + u0) * scale_w
            v = (v_distort * gama2 + v0) * scale_h
            return u.flatten(), v.flatten()

        elif model == "pinhole":
            intr = params['intrinsic']
            f_x, f_y = intr[0, 0], intr[1, 1]
            c_x, c_y = intr[0, 2], intr[1, 2]
            
            # 基础投影
            u = (pc_cam[:, 0] * f_x / (pc_cam[:, 2] + 1e-6)) + c_x
            v = (pc_cam[:, 1] * f_y / (pc_cam[:, 2] + 1e-6)) + c_y
            
            # 如果有畸变参数且需要处理 (这里可以添加 pinhole 畸变逻辑，如果需要的话)
            # 目前直接缩放
            return u * scale_w, v * scale_h
        
        else:
            raise ValueError(f"Unsupported camera model: {model} for camera")


    def sample_from_maps(self, points, batch, feature_maps, cam_params, extrinsics, img_target_size, raw_imgs=None, raw_labels=None):
        with torch.inference_mode():
            B, V, Dim, PH, PW = feature_maps.shape
            # 使用目标尺寸作为边界检查标准            
            # 预分配：point_features 存最终结果，hit_counts 记录每个点被多少个视角覆盖
            point_features = torch.zeros((points.shape[0], Dim), device=points.device, dtype=points.dtype)
            # 用来随机化的 Buffer：记录每个点来自各个视角的候选 (N, V, Dim)
            all_candidates = torch.zeros((points.shape[0], V, Dim), device=points.device, dtype=points.dtype)
            visible_mask = torch.zeros((points.shape[0], V), dtype=torch.bool, device=points.device)

            # 检查并修正 intrinsics 和 extrinsics 的维度
            if extrinsics.dim() == 3: extrinsics = extrinsics.view(B, V, 4, 4)

            for b in range(B):
                H_img, W_img = img_target_size[b][0], img_target_size[b][1]

                b_mask = (batch == b)
                if not b_mask.any(): continue

                # 1. 提取点并强制规范维度
                b_points = points[b_mask].view(-1, 3).float()

                # 准备 Label (如果提供了)
                if hasattr(self, 'vis') and self.vis is not None:
                    b_labels = raw_labels[b_mask] if raw_labels is not None else torch.zeros(len(b_points))
                
                # 2. 构造齐次坐标
                b_points_homo = torch.cat([b_points, torch.ones((b_points.shape[0], 1), device=b_points.device, dtype=b_points.dtype)], dim=1)
                b_global_indices = torch.where(b_mask)[0]
                
                for v in range(V):
                    # 此时 extrinsics[b, v] 保证是 [4, 4] 矩阵
                    pc_cam_homo = (extrinsics[b, v] @ b_points_homo.T).T
                    # 转换为3D点：移除最后一维（应该是1）
                    pc_cam = pc_cam_homo[:, :3] / pc_cam_homo[:, 3:4]  # 归一化
                    
                    depth = pc_cam[:, 2]
                    u, v_coord = self.project_points_unified(pc_cam, cam_params[b][v], (H_img, W_img))

                    valid = (depth > 0.1) & (u >= 0) & (u < W_img) & (v_coord >= 0) & (v_coord < H_img)
                    if not valid.any(): continue
                    
                    # 注意：这里使用 self.patch_size 将坐标映射到 feature map 尺寸
                    u_p = (u[valid] / self.patch_size).long().clamp(0, PW - 1)
                    v_p = (v_coord[valid] / self.patch_size).long().clamp(0, PH - 1)
                    
                    # 获取特征 [N_valid, Dim]
                    sampled_feats = feature_maps[b, v, :, v_p, u_p].transpose(0, 1)

                    # 填充候选矩阵
                    valid_global_idx = b_global_indices[valid]
                    all_candidates[valid_global_idx, v] = sampled_feats
                    visible_mask[valid_global_idx, v] = True
                    
                    if hasattr(self, 'vis') and self.vis is not None:
                        current_view_point_feats = torch.zeros((len(b_points), Dim), device=points.device)
                        current_view_point_feats[valid] = sampled_feats
                        if raw_imgs is not None:
                            if raw_imgs.dim() == 4: # [B*V, C, H, W]
                                image_idx = b * V + v
                                image = raw_imgs[image_idx] if image_idx < raw_imgs.shape[0] else None
                            elif raw_imgs.dim() == 5: # [B, V, C, H, W]
                                image = raw_imgs[b, v]
                            else: continue

                            if image is not None:
                                self.vis.process_frame(
                                    points = b_points,
                                    labels = b_labels,
                                    image = image,
                                    dino_map = feature_maps[b, v],
                                    dino_points_feat = current_view_point_feats,
                                    proj_u = u[valid],
                                    proj_v = v_coord[valid],
                                    proj_depth = depth[valid],
                                    proj_labels = b_labels[valid]
                                )
                
            # 张量化随机选择
            random_weights = torch.rand(visible_mask.shape, device=points.device) * visible_mask
            selected_view_idx = random_weights.argmax(dim=1)
            
            # 提取最终特征
            row_indices = torch.arange(points.shape[0], device=points.device)
            point_features = all_candidates[row_indices, selected_view_idx]

            # 为当前batch绘制最终点云特征
            if hasattr(self, 'vis') and self.vis is not None:
                for b in range(B):
                    b_mask = (batch == b)
                    if not b_mask.any(): continue
                    b_points = points[b_mask].view(-1, 3)
                    b_features = point_features[b_mask]
                    b_visible_any = visible_mask[b_mask].any(dim=1)
                    if b_visible_any.any():
                        self.vis.save_final_point_features(
                            points=b_points[b_visible_any],
                            features=b_features[b_visible_any],
                            batch_idx=b
                        )

        return point_features
