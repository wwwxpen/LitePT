import os
import subprocess
from multiprocessing import Pool, Manager
import argparse
import json

def get_pcd_files(seg_folder_path):
    if not os.path.isdir(seg_folder_path):
        print(f"Directory {seg_folder_path} does not exist.")
        exit(1)
    pcd_files = []
    for root, dirs, files in os.walk(seg_folder_path):
        for file in files:
            if file.endswith(".pcd"):
                pcd_files.append(os.path.join(root, file))
    return pcd_files

def check_condition(pcd_path, invalid_pcd_list, no_delete_pcd):
    # output = subprocess.run(["python", "deploy/check_if_valid_AL_refine.py", "--seged_pcd_path", pcd_path], capture_output=True, text=True).stdout.strip()
    return_code = subprocess.run(["python", "deploy/check_if_valid_AL_refine.py", "--seged_pcd_path", pcd_path]).returncode
    # if output == 'False':
    if return_code == 1:
        print(f"Not valid detected: {pcd_path}")
        if not no_delete_pcd:
            os.remove(pcd_path)
        invalid_pcd_list.append(pcd_path)

def process_batch(batch, invalid_pcd_list, no_delete_pcd):
    with Pool(processes=32) as pool:
        for pcd in batch:
            pool.apply_async(check_condition, args=(pcd, invalid_pcd_list, no_delete_pcd))
        pool.close()
        pool.join()

def main(seg_folder_path, output_file_path, no_delete_pcd):
    pcd_files = get_pcd_files(seg_folder_path)
    batch_size = 64

    # 使用 Manager 来创建一个可以在多进程间共享的列表
    with Manager() as manager:
        invalid_pcd_list = manager.list()
        for i in range(0, len(pcd_files), batch_size):
            batch = pcd_files[i:i + batch_size]
            process_batch(batch, invalid_pcd_list, no_delete_pcd)

        # 将共享列表转换为普通列表并写入文件
        with open(output_file_path, 'w') as f:
            for pcd in invalid_pcd_list:
                f.write(pcd + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PCD files in a given directory.")
    parser.add_argument("--seg_folder_path", type=str, help="Path to the directory containing PCD files.")
    parser.add_argument("--output_file_path", type=str, help="Path to the output file.")
    parser.add_argument("--no_delete_pcd", action='store_true', default=False, help="Not delete bad PCD files.")
    args = parser.parse_args()
    print("start check_AL_valid, no_delete_pcd = ", args.no_delete_pcd)

    # 调用 main 函数
    main(args.seg_folder_path, args.output_file_path, args.no_delete_pcd)