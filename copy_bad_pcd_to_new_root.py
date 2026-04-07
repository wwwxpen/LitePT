import os
import shutil
import argparse
import sys

def move_pcd_files(data_root, bad_pcd_txt_path, data_root_reinfer):
    # 检查 bad_pcd.txt 文件是否存在
    if not os.path.exists(bad_pcd_txt_path):
        print(f"错误：文件 {bad_pcd_txt_path} 不存在。")
        return True

    # 如果目标路径存在，则删除并重新创建
    if os.path.exists(data_root_reinfer):
        shutil.rmtree(data_root_reinfer)
    os.makedirs(data_root_reinfer, exist_ok=True)

    # 读取 bad_pcd.txt 文件
    with open(bad_pcd_txt_path, 'r') as file:
        lines = file.readlines()

    # 检查文件是否为空
    if not lines:
        print("no bad pcd to process")
        return True

    for line in lines:
        line = line.strip()  # 去除可能的换行符等多余字符
        if not line:
            continue
        # 解析文件名和scene
        file_name_with_ext = line.split('/')[-1]
        file_name_without_ext = os.path.splitext(file_name_with_ext)[0]
        parts = file_name_without_ext.split('_')
        file_name = '_'.join(parts[:-2]) + '.pcd'
        scene = parts[-1]
        # 构建源文件路径
        source_file_path = os.path.join(data_root, scene, 'lidarFusion_pcd', file_name)
        # img路径
        img_file_name = '_'.join(parts[:-2]) + '.jpg'

        source_tvFront_path = os.path.join(data_root, scene, 'tvFront_raw', img_file_name)
        source_tvLeft_path = os.path.join(data_root, scene, 'tvLeft_raw', img_file_name)
        source_tvRear_path = os.path.join(data_root, scene, 'tvRear_raw', img_file_name)
        source_tvRight_path = os.path.join(data_root, scene, 'tvRight_raw', img_file_name)

        img_path_exist = os.path.exists(source_tvFront_path)

        # 构建目标文件路径
        target_dir = os.path.join(data_root_reinfer, scene, 'lidarFusion_pcd')
        os.makedirs(target_dir, exist_ok=True)  # 确保目标目录存在
        target_file_path = os.path.join(target_dir, file_name)
        # 构建img文件路径
        if img_path_exist:
            target_tvFront_dir = os.path.join(data_root_reinfer, scene, 'tvFront_raw')
            target_tvLeft_dir = os.path.join(data_root_reinfer, scene, 'tvLeft_raw')
            target_tvRear_dir = os.path.join(data_root_reinfer, scene, 'tvRear_raw')
            target_tvRight_dir = os.path.join(data_root_reinfer, scene, 'tvRight_raw')
            os.makedirs(target_tvFront_dir, exist_ok=True)
            os.makedirs(target_tvLeft_dir, exist_ok=True)
            os.makedirs(target_tvRear_dir, exist_ok=True)
            os.makedirs(target_tvRight_dir, exist_ok=True)
            target_tvFront_path = os.path.join(target_tvFront_dir, img_file_name)
            target_tvLeft_path = os.path.join(target_tvLeft_dir, img_file_name)
            target_tvRear_path = os.path.join(target_tvRear_dir, img_file_name)
            target_tvRight_path = os.path.join(target_tvRight_dir, img_file_name)

        # 拷贝文件
        if os.path.exists(source_file_path):
            shutil.copy(source_file_path, target_file_path)
        else:
            print(f"警告：文件 {file_name} 在源路径 {source_file_path} 中不存在，跳过拷贝。")
        # 拷贝img
        if img_path_exist:
            shutil.copy(source_tvFront_path, target_tvFront_path)
            shutil.copy(source_tvLeft_path, target_tvLeft_path)
            shutil.copy(source_tvRear_path, target_tvRear_path)
            shutil.copy(source_tvRight_path, target_tvRight_path)

    # 拷贝data.json
    source_data_json_path = os.path.dirname(data_root) + "/data.json"
    target_data_json_path = os.path.dirname(data_root_reinfer) + "/data.json"
    if os.path.exists(source_data_json_path):
        shutil.copy(source_data_json_path, target_data_json_path)
    else:
        print(f"警告：文件 {img_file_name} 在源路径 {source_data_json_path} 中不存在，跳过拷贝。")    

    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move pcd files based on bad_pcd.txt")
    parser.add_argument("--data_root", type=str, help="Root path of the data")
    parser.add_argument("--bad_pcd_txt_path", type=str, help="Path to the bad_pcd.txt file")
    parser.add_argument("--data_root_reinfer", type=str, help="Path to move the files to")
    args = parser.parse_args()

    result = move_pcd_files(args.data_root, args.bad_pcd_txt_path, args.data_root_reinfer)
    if result:
        sys.exit(0)  # 返回 0 表示 True
    else:
        sys.exit(1)  # 返回 1 表示 False
    print(f"返回值: {result}")