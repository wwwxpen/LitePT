#!/usr/bin/env python3
import os
import open3d as o3d
import sys
import argparse

def check_top_lidar(data_path):
    """
    检查点云数据中是否包含0号激光雷达的数据
    
    Args:
        data_path: 数据根目录路径
        
    Returns:
        int: 1表示包含0号激光雷达数据，0表示不包含
    """
    samples_path = os.path.join(data_path, "samples")
    
    if not os.path.exists(samples_path):
        return 1
    
    # 遍历根文件夹下的所有子文件夹
    for subdir in os.listdir(samples_path):
        subdir_path = os.path.join(samples_path, subdir)
        lidar_fusion_pcd_path = os.path.join(subdir_path, "lidarFusion_pcd")
        
        if os.path.exists(lidar_fusion_pcd_path):
            # 查找第一个pcd文件
            for file in os.listdir(lidar_fusion_pcd_path):
                if file.endswith('.pcd'):
                    pcd_path = os.path.join(lidar_fusion_pcd_path, file)
                    try:
                        # 读取点云文件
                        pcd = o3d.t.io.read_point_cloud(pcd_path)
                        
                        # 检查点云中是否包含lidarId或ring字段
                        if "lidarId" in pcd.point:
                            rings = set(pcd.point['lidarId'].numpy().flatten())
                        elif "ring" in pcd.point:
                            rings = set(pcd.point['ring'].numpy().flatten())
                        else:
                            # 如果没有找到相关字段，继续查找下一个文件
                            continue
                        
                        # 检查0是否在rings中
                        return 1 if 0 in rings else 0
                            
                    except Exception as e:
                        # 如果读取失败，继续尝试下一个文件
                        continue
    
    # 如果没有找到有效的pcd文件，返回1
    return 1

def main():
    parser = argparse.ArgumentParser(description='检查点云数据是否包含0号激光雷达')
    parser.add_argument('--data_path', required=True, help='数据根目录路径')
    
    args = parser.parse_args()
    
    try:
        result = check_top_lidar(args.data_path)
        # 直接通过退出码返回结果，不输出任何内容
        sys.exit(result)
    except Exception as e:
        # 出错时返回1（默认包含0号激光雷达，更安全）
        sys.exit(1)

if __name__ == "__main__":
    main()
