import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import timm # 【修改】使用 timm 替代 torch.hub

try:
    import open3d as o3d
except ImportError:
    o3d = None

class DINOFeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vitl14', local_weight_path=None, frozen=True):
        super().__init__()
        
        self.num_register_tokens = 0 if 'dinov2' in model_name else 4 # 根据模型自动设置 register tokens 数量
        self.patch_size = 14 if 'dinov2' in model_name else 16  # 根据模型自动设置 patch size

        # 【修改】将 DINO 官方名称映射到 timm 名称
        # timm 的命名规则略有不同，.lvd142m 后缀代表官方权重
        name_map = {
            # DINOv2 模型（patch size 14）
            'dinov2_vitl14': 'vit_large_patch14_dinov2.lvd142m',
            'dinov2_vitb14': 'vit_base_patch14_dinov2.lvd142m',
            'dinov2_vits14': 'vit_small_patch14_dinov2.lvd142m',
            # DINOv3 模型（patch size 16，带寄存器）
            "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
            "dinov3_vitsp16": "vit_small_plus_patch16_dinov3.lvd1689m",
            "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
            "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
        }
        timm_name = name_map.get(model_name, model_name)

        print(f"[DITR] Loading DINO from timm: {timm_name}")
        # local_weight_path 由外部传入配置文件；为 None 则使用 timm 下载/预训练
        if local_weight_path is None:
            # 未提供 local path -> 使用 timm 的 pretrained（下载）
            pretrained_flag = True
            ckpt_path = ""
            print(f"[DITR] No local weight path provided -> will use timm pretrained download for {timm_name}")
        else:
            # 明确提供了本地路径：必须存在，否则报错（避免意外下载/版本不一致）
            if not os.path.exists(local_weight_path):
                raise FileNotFoundError(f"[DITR] local_weight_path set to '{local_weight_path}' but file not found. "
                                        "Please provide a valid path or set local_weight_path=None to use timm pretrained.")
            pretrained_flag = False
            ckpt_path = local_weight_path
            print(f"[DITR] Using local DINO weights at: {local_weight_path}")


        # dynamic_img_size=True 允许处理不同分辨率的图片输入
        self.dino = timm.create_model(
            timm_name, 
            pretrained=pretrained_flag,     # 关闭自动下载
            checkpoint_path=ckpt_path,      # 指定本地路径
            num_classes=0, 
            dynamic_img_size=True 
        )
        
        if frozen:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
            
    def forward(self, images):
        # 兼容处理 5D 和 4D 输入
        if images.dim() == 5:
            # images: [B, N_views, 3, H, W] -> Flatten -> [B*N, 3, H, W]
            b, n, c, h, w = images.shape
            x = images.view(b * n, c, h, w)
        else:
            raise ValueError(f"Unexpected images shape: {images.shape}. Expected 4 or 5 dimensions.")
        
        with torch.inference_mode():  # 比 torch.no_grad() 更快
            # 【修改】timm 的 forward_features 返回 [Batch, N_tokens, Dim]
            # 包含了 CLS token 和 Patch tokens
            out = self.dino.forward_features(x)
            
            # 这里的 out 包含了 CLS token (index 0)
            # 我们只需要 patch tokens (index 1:)
            # 另外 DINOv2 有些变体可能有 register tokens，但标准版通常只有 CLS
            # 标准 ViT: out[:, 1:, :]
            patch_features = out[:, (1+self.num_register_tokens):, :]
            
        dim = patch_features.shape[-1]
        p_h, p_w = h // self.patch_size, w // self.patch_size
        
        # Reshape 回空间维度 
        # [B*N, H*W, Dim] -> [B, N, H, W, Dim] -> [B, N, Dim, H, W]
        feature_map = patch_features.view(b, n, p_h, p_w, dim)
        feature_map = feature_map.permute(0, 1, 4, 2, 3) 
        return feature_map

class DITRInjector(nn.Module):
    def __init__(self, dino_model, debug=False, output_dir="debug_ditr"):
        super().__init__()
        self.dino = dino_model
        self.debug = debug
        self.output_dir = output_dir
        if debug:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[DITR] Debug mode enabled. Visualization will be saved to {output_dir}")

    def viz_projection(self, img_tensor, u, v, fname):
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3,1,1)
        img = img_tensor * std + mean
        img = img.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        # 随机画一些点，防止过密
        if len(u) > 500:
            idx = torch.randperm(len(u))[:500]
            u, v = u[idx], v[idx]
        plt.scatter(u.cpu().numpy(), v.cpu().numpy(), s=2, c='red', alpha=0.8)
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, fname), bbox_inches='tight', pad_inches=0)
        plt.close()

    def viz_dino_pca(self, feature_map, fname):
        c, h, w = feature_map.shape
        reshaped = feature_map.view(c, -1).T.float().cpu().numpy()
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca_f = pca.fit_transform(reshaped)
        pca_f = (pca_f - pca_f.min(0)) / (pca_f.max(0) - pca_f.min(0) + 1e-6)
        pca_img = pca_f.reshape(h, w, 3)
        plt.imsave(os.path.join(self.output_dir, fname), pca_img)

    def viz_pcd_features(self, points, features, fname):
        if o3d is None or features.sum() == 0: return
        
        if len(points) > 200000:
            mask = torch.randperm(len(points))[:200000]
            points = points[mask]
            features = features[mask]
            
        points_np = points.cpu().numpy()
        feats_np = features.float().cpu().numpy()
        
        from sklearn.decomposition import PCA
        valid_mask = np.abs(feats_np).sum(1) > 0
        if valid_mask.sum() < 10: return
        
        colors = np.zeros((len(points_np), 3))
        colors[~valid_mask] = [0.5, 0.5, 0.5] 
        
        pca = PCA(n_components=3)
        pca_c = pca.fit_transform(feats_np[valid_mask])
        pca_c = (pca_c - pca_c.min(0)) / (pca_c.max(0) - pca_c.min(0) + 1e-6)
        colors[valid_mask] = pca_c
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(self.output_dir, fname), pcd)