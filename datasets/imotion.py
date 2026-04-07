import os
import numpy as np
from collections.abc import Sequence
import torch
import json
import cv2
import glob
from scipy.spatial.transform import Rotation
from turbojpeg import TurboJPEG
from concurrent.futures import ThreadPoolExecutor

from .builder import DATASETS
from .defaults import DefaultDataset

from da_binds import io as da_io

from turbojpeg import TurboJPEG, TJPF_RGB, TJSAMP_420
import cv2

@DATASETS.register_module()
class ImotionDataset(DefaultDataset):
    # 严格照搬 nuscenes.py 的类变量定义
    CAMERA_TYPES = ["tvFront", "tvLeft", "tvRear", "tvRight"] # 对应 4 个鱼眼
    
    def __init__(self,
                 split=(),
                 data_root="data/imotion",
                 transform=None,
                 test_cfg=None,
                 load_camera=False,
                 img_size=(448, 560),
                 test_mode=None,
                 validation_mode=False,
                 ignore_index=-1,
                 denoise=False,
                 **kwargs):
        self.data_root = data_root
        self.load_camera = load_camera
        self.test_mode = test_mode
        self.validation_mode = validation_mode
        self.ignore_index = ignore_index
        self.denoise = denoise
        if self.load_camera:
            self.cam_names = self.CAMERA_TYPES
            self.img_size = img_size
            print(f"Initialized ImotionDataset with load_camera={self.load_camera}, img_size={self.img_size}, test_mode={self.test_mode}, validation_mode={self.validation_mode}, ignore_index={self.ignore_index}, denoise={self.denoise}")
            # DINOv2 官方推荐的归一化参数
            # self.transform_img = T.Compose([
            #     T.Resize(self.img_size), # Resize ((h, w))
            #     T.ToTensor(),
            #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])
            # 提前转为 numpy 格式，方便在线程中直接计算
            self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        super().__init__(split=split, data_root=data_root, transform=transform, test_mode=test_mode, test_cfg=test_cfg, **kwargs)

    def _get_json_list(self, json_path):
        """
        根据样例 JSON 结构解析数据列表。
        适配字段: pcd_path, json_path, 以及 tvFront_path 等四路图片路径。
        """
        # 检查索引文件是否存在
        if not os.path.exists(json_path):
            return []

        with open(json_path, 'r') as f:
            infos = json.load(f)

        data_list = []
        for info in infos:
            # 1. 基础 PCD 路径必选
            data_item = {
                "pcd_path": info.get("pcd_path")
                # "full_name": info.get("full_name")
            }
            # 2. 根据 load_camera 决定是否加载视觉相关路径
            if self.load_camera:
                # 与 self.cam_names 一致
                img_paths = [info.get(f"{cam}_path") for cam in self.cam_names]
                # 获取当前条目指定的 data.json 路径
                calib_path = info.get("json_path")
                # 更新字典，包含视觉全量信息
                data_item.update({
                    "img_paths": img_paths,
                    "json_path": calib_path
                })
            data_list.append(data_item)

        return data_list

    def get_data_list(self):
        # 推理模式且非验证模式（无gt）：直接遍历 samples/scene*/lidarFusion_pcd/*.pcd
        if self.test_mode and not self.validation_mode:
            print("data_root =", self.data_root)
            data_list = []
            # denoise去噪模式，直接返回data_root下所有pcd，无需图像
            if self.denoise:
                pcds = sorted(glob.glob(os.path.join(self.data_root, "*.pcd")))
                for p in pcds:
                    data_list.append({"pcd_path": p})
                return data_list  
            # 正常模式
            scene_paths = sorted(glob.glob(os.path.join(self.data_root, "*scene*")))
            for scene in scene_paths:
                pcd_dir = os.path.join(scene, "lidarFusion_pcd")
                if os.path.isdir(pcd_dir):
                    pcds = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
                    for p in pcds:
                        if self.load_camera:
                            file_name = os.path.splitext(os.path.basename(p))[0]
                            # 推理模式规则：带 _raw 的文件夹
                            img_paths = [os.path.join(scene, f"{cam}_raw", f"{file_name}.jpg") for cam in self.cam_names]
                            # 兼容性检查 (.jpg vs .png)
                            img_paths = [img if os.path.exists(img) else img.replace(".jpg", ".png") for img in img_paths]
                            
                            data_list.append({
                                "pcd_path": p,
                                "img_paths": img_paths,
                                "json_path": os.path.join(os.path.dirname(self.data_root), "data.json"),
                                "scene_path": scene
                            })
                        else:
                            data_list.append({"pcd_path": p})
            return data_list            
        # 训练模式：根据索引文件读取路径列表
        else:
            train_data = self._get_json_list("/mlp/data_loop/workspace_wxp/seg_GT/seg_GT_pcds_4train_clean_data/seg_GT_info_4train_clean_badcase_complete.json")
            val_data = self._get_json_list("/mlp/data_loop/workspace_wxp/seg_GT/seg_GT_pcds_4train_clean_data/seg_GT_info_4val_clean_badcase_complete.json")
            test_data = self._get_json_list("/mlp/data_loop/workspace_wxp/seg_GT/seg_GT_pcds_4train_clean_data/seg_GT_info_4test_clean_badcase_complete.json")
            print("data_root =", self.data_root)
            print(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples, {len(test_data)} test samples from JSON index files.")

            split_dict = {"train": train_data, "val": val_data, "test": test_data, }
            print(f"Loaded data splits with counts: " + ", ".join([f"{k}: {len(v)}" for k, v in split_dict.items()]))

            if isinstance(self.split, str):
                split_data_list = split_dict[self.split]
            elif isinstance(self.split, Sequence):
                split_data_list = []
                for s in self.split:
                    split_data_list.extend(split_dict[s])
            else:
                raise NotImplementedError
            return split_data_list

    def _get_camera_params(self, calib_data, cam_name):
        """
        统一解析 data-old.json 和 data-new.json
        返回: params_dict (包含内参模型参数), extrinsic (4x4 矩阵)
        """
        # 1. 确定使用的模型
        default_models = calib_data.get("default_cameramodel", {})
        # 如果在 default_cameramodel 中定义了模型，则使用它，否则默认为 pinhole
        model_type = default_models.get(cam_name, "pinhole")
        
        # 2. 获取对应的配置块
        cam_cfg = calib_data.get(cam_name, {})
        if model_type in cam_cfg:
            # data-new.json 结构: { cam_name: { pinhole: {...}, mei: {...} } }
            cfg = cam_cfg[model_type]
        else:
            # data-old.json 结构或 data-new 的平铺结构
            cfg = cam_cfg

        # 3. 解析外参 (Rotation + Translation)
        extrinsic = np.eye(4)
        # 四元数格式转换: 从 wxyz 转为 xyzw
        quat_wxyz = np.array(cfg["rotation"])  # imotion是这个格式 [w, x, y, z]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]  # [x, y, z, w]
        # 处理旋转 (Quat -> Matrix)
        extrinsic[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
        extrinsic[:3, 3] = np.array(cfg["translation"])
        # 关键：求逆得到投影矩阵
        extrinsic = np.linalg.inv(extrinsic)

        # 4. 构建参数字典
        params = {"model": model_type}
        
        if model_type == "pinhole":
            # 获取内参矩阵
            params["intrinsic"] = np.array(cfg.get("camera_intrinsic"))
            # data-new.json 的 pinhole 可能带有 distortion
            params["distortion"] = np.array(cfg.get("distortion", []))
            # 记录原始分辨率用于缩放计算
            res = cfg.get("resolution", "1920*1536").split("*")
            params["base_size"] = [float(res[1]), float(res[0])] # [H, W]
            # base_size = np.array([h, w]) # [H, W]

        elif model_type == "mei":
            m = cfg # mei 的参数直接在这一层
            params.update({
                "ksi": m["ksi"],
                "k": m["k"],      # k1, k2, k3, k4...
                "center": m["center"], # u0, v0
                "gama": m["gama"],     # gama1, gama2
            })
            res = m.get("resolution", "1920*1536").split("*")
            params["base_size"] = [float(res[1]), float(res[0])]
            
        else:
            raise ValueError(f"Unsupported camera model: {model_type} for camera {cam_name}")

        return params, extrinsic

    def _load_image_turbo(self, img_path):
        """复原逻辑：TurboJPEG(RGB) + Resize + Numpy 向量化标准化"""
        with open(img_path, 'rb') as f:
            img = self.jpeg.decode(f.read(), pixel_format=TJPF_RGB)
        
        # 1. Resize
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        
        # 2. 标准化 (严格复原：/255 -> 减均值 -> 除标准差)
        img_tensor = (img.astype(np.float32) / 255.0 - self.img_mean) / self.img_std
        
        # 3. HWC -> CHW
        return img_tensor.transpose(2, 0, 1)

    def get_data(self, idx):
        item = self.data_list[idx % len(self.data_list)]
        data_path = item["pcd_path"]

        # PCD 读取
        if self.test_mode and not self.validation_mode:
            scan = da_io.read_pcd_xyzi_structured(data_path)
            coord=scan[:, :3]
            segment = np.ones((scan.shape[0],), dtype=np.int64) * self.ignore_index
        else:
            scan, segment = da_io.read_pcd_xyzis_structured(data_path)
            coord=scan[:, :3]
            segment = segment.reshape(-1).astype(np.int64)

        if self.load_camera:    
            data_dict = dict(
                coord=scan[:, :3],
                color=coord.copy(),  # 【新增】 使用原始坐标作为颜色输入，便于同步下采样
                strength=scan[:, -1].reshape([-1, 1]) / 255.0, 
                segment=segment, 
                name=os.path.splitext(os.path.basename(data_path))[0],
            )
        else:
            data_dict = dict(
                coord=scan[:, :3],
                strength=scan[:, -1].reshape([-1, 1]) / 255.0, 
                segment=segment, 
                name=os.path.splitext(os.path.basename(data_path))[0],
            )

        # 2. 视觉模态逻辑 (仅在 load_camera=True 时加载TurboJPEG和线程池)
        # 视觉模态读取
        if self.load_camera and "img_paths" in item:
            #---  懒加载TurboJPEG,ThreadPoolExecutor ---
            if not hasattr(self, 'jpeg'): self.jpeg = TurboJPEG()
            if not hasattr(self, 'executor'):
                self.executor = ThreadPoolExecutor(max_workers=len(self.cam_names))

            # 加载标定 (路径已在 get_data_list 中确定)
            with open(item["json_path"], 'r') as f:
                calib_data = json.load(f)

            cam_params_list, extrinsics = [], []
            for cam in self.cam_names:
                p_dict, ext = self._get_camera_params(calib_data, cam)
                cam_params_list.append(p_dict)
                extrinsics.append(ext)

            # 直接使用 list_list 里的 img_paths
            imgs = list(self.executor.map(self._load_image_turbo, item["img_paths"]))
            data_dict.update(dict(
                imgs=np.array(imgs).astype(np.float32),
                cam_params=cam_params_list, # 这是一个 list of dicts，包含每个摄像头的参数
                extrinsics=np.array(extrinsics).astype(np.float32),
                img_target_size=np.array([self.img_size[0], self.img_size[1]]).astype(np.float32)
            ))
        return data_dict
