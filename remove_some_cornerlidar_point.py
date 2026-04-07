import open3d as o3d
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool
import time
import argparse
import re
import json
from functools import partial
from tqdm import tqdm
from da_binds import io as da_io

def load_data_json(data_path):
    json_path = os.path.join(data_path, 'data.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    translation = data['lidarTop']['body']['translation']
    print(f"LidarTop position: {translation}")
    return translation

def process_pcd_file(seged_pcd_path, sample_pcd_files, lidartop_pos):
    # Read pcdA
    # pcdA = o3d.t.io.read_point_cloud(seged_pcd_path)
    cloud = da_io.read_pcd_cloud_structured(seged_pcd_path)
  
    x0, y0, z0 = lidartop_pos
    # Filter pcdA
    # positions_A = pcdA.point['positions'].numpy()
    # seg_labels_A = pcdA.point['segLabel'].numpy()

    if not 'segLabel' in cloud.dtype.names or not 'ring' in cloud.dtype.names:
        print("segLabel or ring not in pcd, do not remove corner lidar pt")
        return False
    
    positions = cloud['xyzi'][:, :3]  # 提取xyz
    seg_labels = cloud['segLabel']
    if 'lidarId' in cloud.dtype.names:
        lidarIds = cloud['lidarId']
    else:
        lidarIds = cloud['ring']
    keep_indices = []
    for i, (pos, seg_label, lidar_id) in enumerate(zip(positions, seg_labels, lidarIds)):
        x1, y1, z1 = pos
        # Filter by lidarId and segLabel
        if seg_label not in [2, 3, 9, 10, 11, 12, 13, 16, 25]:
            if lidar_id == 0:
                keep_indices.append(i)
            elif seg_label==23 or seg_label==14 or seg_label==27 or seg_label==18:
                if x1*x1+y1*y1 < 400:  # 20m范围内的23地面障碍物/14锥桶/27杆状物/18栅栏保留，怕不保留轮档等低矮小障碍物太稀疏
                    keep_indices.append(i)
            else:
                # Calculate degree
                dx = x1 - x0
                dy = y1 - y0
                dz = z1 - z0
                horizontal_dist = np.sqrt(dx*dx + dy*dy)
                degree = np.degrees(np.arctan2(dz, horizontal_dist))
                if not (-22 <= degree <= 15):
                    keep_indices.append(i)
        else:
                keep_indices.append(i)
    # Apply filtering and overwrite original file
    # filtered_pcd = pcdA.select_by_index(keep_indices)
    filtered_cloud = cloud[keep_indices]
    base, ext = os.path.splitext(seged_pcd_path)
    tmp_pcd_path = f"{base}_tmp{ext}"  # 例如：path/file_tmp.pcd
    # o3d.t.io.write_point_cloud(tmp_pcd_path, filtered_pcd)
    da_io.save_cloud_structure(tmp_pcd_path, filtered_cloud)
    os.remove(seged_pcd_path)
    os.rename(tmp_pcd_path, seged_pcd_path)
    # print(f"Successfully processed and overwritten: {seged_pcd_path}")
    return True

def process_pcd_files_in_subfolders(data_path, seg_folder_path):
    """
    遍历seged_pcd下的所有pcd文件,并对其中的PCD文件执行process_pcd_file函数。
    data_path (str): 根文件夹路径。
    """
    # 获取CPU数量
    cpu_count = multiprocessing.cpu_count()
    # 计算进程数量
    num_processes = max(int(cpu_count * 2 / 3), 20)
    print("Use num_processes = ", num_processes)
    
    # 遍历samples文件夹下的所有子文件夹
    sample_pcd_files = []
    samples_path = os.path.join(data_path, "samples")
    if os.path.exists(samples_path):
        for subdir in os.listdir(samples_path):
            subdir_path = os.path.join(samples_path, subdir)
            lidar_fusion_pcd_path = os.path.join(subdir_path, "lidarFusion_pcd")
            if os.path.exists(lidar_fusion_pcd_path):
                for file in os.listdir(lidar_fusion_pcd_path):
                    if file.endswith(".pcd"):
                        sample_pcd_files.append(os.path.join(lidar_fusion_pcd_path, file))
    try:
        lidartop_pos = load_data_json(data_path)
    except Exception as e:
        print(f"An unexpected error occurred while loading the JSON file: {str(e)}, filter seged pcd stopped")
        return

    # Get all segmented pcd files (pcdA files)
    seged_pcd_files = []
    if os.path.exists(seg_folder_path):
        for file in os.listdir(seg_folder_path):
            if file.endswith(".pcd"):
                seged_pcd_files.append(os.path.join(seg_folder_path, file))
    
    # filter seged_pcds
    # for seged_pcd_path in seged_pcd_files:
    #     process_pcd_file(seged_pcd_path, sample_pcd_files, lidartop_pos)
    with multiprocessing.get_context("spawn").Pool(processes=num_processes) as pool:
        pbar = tqdm(total=len(seged_pcd_files), desc="Processing seged PCD files: rm some corner lidar pt")
        def update_pbar(*_):
            pbar.update(1)
        for seged_pcd_path in seged_pcd_files:
            pool.apply_async(
                func=process_pcd_file, 
                args=(seged_pcd_path, sample_pcd_files, lidartop_pos),
                callback=update_pbar
            )
        pool.close()
        pool.join()
        pbar.close()
    print(f"✅ 所有seged pcd处理已完成, 共处理 {len(seged_pcd_files)} 个文件")  

# 示例调用
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter PCD files in seged_pcds.")
    parser.add_argument("--data_path", type=str, help="Root folder path containing PCD files.")
    parser.add_argument("--seg_folder_path", type=str, help="seged folder path containing seged PCD files.")
    args = parser.parse_args()
    print("data_path = ", args.data_path)
    print("seg_folder_path = ", args.seg_folder_path)
    print("start rm some corner lidar point in seged pcds")
    process_pcd_files_in_subfolders(args.data_path, args.seg_folder_path)