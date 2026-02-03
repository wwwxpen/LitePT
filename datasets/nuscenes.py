import os
import numpy as np
from collections.abc import Sequence
import pickle

from .builder import DATASETS
from .defaults import DefaultDataset

from PIL import Image
import open3d as o3d
import torch
import torchvision.transforms as T  # 【新增】


@DATASETS.register_module()
class NuScenesDataset(DefaultDataset):
    # 【新增】 定义相机顺序，确保 batch 堆叠时顺序一致
    CAMERA_TYPES = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    def __init__(self, sweeps=10, ignore_index=-1, 
                 # 【新增】 DITR 相关参数
                load_camera=False,
                img_size=(378, 672),
                 **kwargs):
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)

        self.load_camera = load_camera
        self.img_size = img_size
        # DINOv2 官方推荐的归一化参数
        if self.load_camera:
            self.transform_img = T.Compose([
                T.Resize(self.img_size), # Resize ((h, w))
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]

        # 这一步非常关键！必须在 augmentation 之前保存。用于后续可视化
        # Pointcept 的 GridSample 会自动处理 'origin_coord' 键，对其进行同步下采样。
        origin_coord = coord.copy()

        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        data_dict = dict(
            coord=coord,

            color=origin_coord,  # 【新增】 使用原始坐标作为颜色输入，便于同步下采样

            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )

        # ======================================================================
        # 【新增】 视觉模态加载逻辑 (参考自 NuScenesImagePointDataset)
        # ======================================================================
        if self.load_camera and "cams" in data:
            imgs = []
            intrinsics = []
            extrinsics = [] # Lidar -> Camera 的变换矩阵
            # print(f"\n=== Processing sample: {self.get_data_name(idx)} ===")
            # print(f"Available cameras in data: {list(data['cams'].keys())}")
            for cam_name in self.CAMERA_TYPES:
                # 1. 获取 Info
                if cam_name not in data["cams"]:
                    # 极其罕见的情况，填充全0或跳过，这里假设数据完整
                    continue
                cam_info = data["cams"][cam_name]
                # print(f"  Camera token: {cam_info.get('sample_token', 'N/A')}")
                # print(f"  Data path: {cam_info.get('data_path', 'N/A')}")
                
                # 2. 读取并处理图片
                img_path = os.path.join(self.data_root, "raw", cam_info["data_path"])
                # 必须转为 RGB，防止有 PNG alpha 通道或者灰度图
                img_pil = Image.open(img_path).convert('RGB') 
                
                # 记录原始尺寸用于缩放内参
                w_orig, h_orig = img_pil.size
                
                # Transform (Resize -> ToTensor -> Normalize)
                img_tensor = self.transform_img(img_pil)
                imgs.append(img_tensor)
                
                # 3. 处理内参 (Intrinsics)
                # 因为图片 Resize 了，内参矩阵也需要相应缩放
                intr = np.eye(3)
                intr[:3, :3] = cam_info["camera_intrinsics"]
                
                # 计算缩放比例
                h_target, w_target = self.img_size
                scale_w = w_target / w_orig
                scale_h = h_target / h_orig
                
                # 调整 fx, cx
                intr[0, 0] *= scale_w
                intr[0, 2] *= scale_w
                # 调整 fy, cy
                intr[1, 1] *= scale_h
                intr[1, 2] *= scale_h
                intrinsics.append(intr)
                # print(f"  Intrinsics (original): {cam_info['camera_intrinsics']}")
                # print(f"  Intrinsics (scaled {scale_w:.3f}x{scale_h:.3f}):\n{intr}")
                
                # 4. 处理外参 (Extrinsics: Lidar -> Camera)
                # Info 中通常给出的是 Sensor(Cam) -> Lidar 的变换
                sensor2lidar = np.eye(4)
                sensor2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
                sensor2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
                # 我们需要 Lidar -> Camera，所以求逆
                lidar2sensor = np.linalg.inv(sensor2lidar)
                extrinsics.append(lidar2sensor)
                # print(f"  sensor2lidar rotation:\n{cam_info['sensor2lidar_rotation']}")
                # print(f"  sensor2lidar translation: {cam_info['sensor2lidar_translation']}")

            # 堆叠并转为 Tensor
            if len(imgs) == 6:
                data_dict['imgs'] = torch.stack(imgs) # [6, 3, H, W]
                data_dict['intrinsics'] = torch.tensor(np.stack(intrinsics), dtype=torch.float32) # [6, 3, 3]
                data_dict['extrinsics'] = torch.tensor(np.stack(extrinsics), dtype=torch.float32) # [6, 4, 4]
            else:
                # 容错处理：如果数据不全，可能会报错，实际 nuScenes 数据通常是完整的
                print(f"Warning: {self.get_data_name(idx)} missing cameras.")

        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map
