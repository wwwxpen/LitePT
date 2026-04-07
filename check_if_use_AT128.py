import open3d as o3d
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool
import time
import argparse
from da_binds import io as da_io

def is_valid_package(package_name):
    # 定义车辆及其对应的日期
    car_dates = {
        "M36T": "0427",
        "E03": "0423",
        "E0Y": "0425",
        "V23": "0424",
        "T22": "0428"
    }

    # 提取日期和车名
    date_part = package_name.split('_')[2].split('-')[0]
    car_name = package_name.split('_')[3].split('-')[0]
    print("date_part = ", date_part, ", car_name = ", car_name)

    # 检查车名是否在字典中
    if car_name not in car_dates:
        return False

    # 比较日期
    start_date = "20250415"
    end_date = "2025" + car_dates[car_name]

    return start_date <= date_part < end_date


def filter_point_cloud_by_ring(input_pcd_path, ring_value_to_remove):
    """
    根据ring属性过滤点云，并替换原PCD文件。
    参数：
    input_pcd_path (str): 输入的PCD文件路径。
    ring_value_to_remove (int): 需要移除的ring值。
    """
    try:
        # 读取PCD文件
        # print("ring filter processing pcd: ", input_pcd_path)
        # pcd = o3d.t.io.read_point_cloud(input_pcd_path)
        # # 假设点云有ring属性，获取点的ring
        # seg_labels = pcd.point["ring"]
        # # 创建一个布尔掩码，去除ring为指定值的点
        # mask = (seg_labels != ring_value_to_remove)
        # # 应用掩码
        # # 获取满足条件的索引
        # indices = np.where(mask)[0]
        # indices_list = indices.tolist()
        # filtered_pcd = pcd.select_by_index(indices_list)
        # # 将处理后的点云保存为新的PCD文件，直接替换原文件
        # o3d.t.io.write_point_cloud(input_pcd_path, filtered_pcd, compressed=True)

        # 使用新的方式读取点云
        cloud = da_io.read_pcd_cloud_structured(input_pcd_path)
        # 检查cloud是否包含lidarId字段
        has_lidarId = 'lidarId' in cloud.dtype.names
        # 根据不同的字段创建过滤掩码
        if has_lidarId:
            # 使用lidarId字段进行过滤
            mask = (cloud['lidarId'][:, 0] != ring_value_to_remove)
        else:
            # 使用ring字段进行过滤
            mask = (cloud['ring'][:, 0] != ring_value_to_remove)
        # 应用掩码过滤点云
        filtered_cloud = cloud[mask]
        # 保存处理后的点云
        da_io.save_cloud_structure(input_pcd_path, filtered_cloud)
    except Exception as e:
        print(f"Error processing {input_pcd_path}: {e}")


def process_pcd_files_in_subfolders(data_path, ring_value_to_remove):
    """
    遍历指定文件夹下所有子文件夹中的lidarFusion_pcd文件夹，并对其中的PCD文件执行filter_point_cloud_by_ring函数。
    参数：
    data_path (str): 根文件夹路径。
    ring_value_to_remove (int): 需要移除的ring值。
    """
    # 获取CPU数量
    cpu_count = multiprocessing.cpu_count()
    # 计算进程数量
    num_processes = max(int(cpu_count * 2 / 3), 20)
    print("Use num_processes = ", num_processes, " to filter lidar ring = 5")
    
    # 遍历根文件夹下的所有子文件夹
    pcd_files = []
    samples_path = os.path.join(data_path, "samples")
    sweeps_path = os.path.join(data_path, "sweeps")
    if os.path.exists(samples_path):
        for subdir in os.listdir(samples_path):
            subdir_path = os.path.join(samples_path, subdir)
            lidar_fusion_pcd_path = os.path.join(subdir_path, "lidarFusion_pcd")
            if os.path.exists(lidar_fusion_pcd_path):
                for file in os.listdir(lidar_fusion_pcd_path):
                    if file.endswith(".pcd"):
                        pcd_files.append(os.path.join(lidar_fusion_pcd_path, file))
    if os.path.exists(sweeps_path):
        for subdir in os.listdir(sweeps_path):
            subdir_path = os.path.join(sweeps_path, subdir)
            lidar_fusion_pcd_path = os.path.join(subdir_path, "lidarFusion_pcd")
            if os.path.exists(lidar_fusion_pcd_path):
                for file in os.listdir(lidar_fusion_pcd_path):
                    if file.endswith(".pcd"):
                        pcd_files.append(os.path.join(lidar_fusion_pcd_path, file))
    
    # 使用多进程处理PCD文件
    start_time = time.time()  # 开始计时

    def callback(result):
        nonlocal processed_count
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"已处理 {processed_count} 个")

    processed_count = 0
    with multiprocessing.get_context("spawn").Pool(processes=num_processes) as pool:
        for file in pcd_files:
            pool.apply_async(func=filter_point_cloud_by_ring, args=(file, ring_value_to_remove), callback=callback)
        pool.close()
        pool.join()

    end_time = time.time()  # 结束计时
    
    # 打印耗时
    elapsed_time = end_time - start_time
    print(f"处理完成，总耗时：{elapsed_time:.2f} 秒, 总数： {processed_count} 个")

# 示例调用
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PCD files in subfolders.")
    parser.add_argument("--data_path", type=str, help="Root folder path containing PCD files.")
    # data_path = "/home/xiaopengwu/Downloads/changepcd/change_ring/direct_change/P_WuHu_20250420-045011_T22-7412_0"  # 替换为你的根文件夹路径
    args = parser.parse_args()
    ring_value_to_remove = 5  # 需要移除的ring值
    print("data_path = ", args.data_path)
    # if is_valid_package(os.path.basename(args.data_path)):
    #     print("Need to regenerate pcd without ring=5: ", os.path.basename(args.data_path))
    #     process_pcd_files_in_subfolders(args.data_path, ring_value_to_remove)
    # else:
    #     print("No Need to regenerate pcd without ring=5: ", os.path.basename(args.data_path))

    # 泊车暂时不使用AT128
    print("Use pcd without ring=5: ", os.path.basename(args.data_path))
    process_pcd_files_in_subfolders(args.data_path, ring_value_to_remove)