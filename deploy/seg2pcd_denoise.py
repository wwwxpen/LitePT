import os
import sys
import argparse
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from pathlib import Path

import numpy as np
# import imo_pcd_reader
from da_binds import io as da_io
import open3d as o3d

# params
parser = argparse.ArgumentParser(description='Point Cloud Segmentation Result Visulization')

parser.add_argument('--result_base_dir', type=str, 
                    default='/your/path/.../result')
parser.add_argument('--data_base_dir', type=str, 
                    default='/your/path/.../dataset/sweeps')
args = parser.parse_args()

def save_pcd_use_noise(base_pcd_pth, infer_segLabels, save_pcd_path):
    # Ensure NumPy arrays have the right shapes and types
    cloud = da_io.read_pcd_cloud_structured(base_pcd_pth)

    xyzi = cloud['xyzi']
    segLabel = cloud['segLabel']

    if segLabel.shape[0] != infer_segLabels.shape[0]:
        print(f"infer_segLabels array must have same shape with input pcd, {segLabel.shape[0]} vs {infer_segLabels.shape[0]}")   
    # change base_pcd_segLabel use infer_segLabels
    count = [0, 0]
    # set type can be delete if infer result is noise, generally should be a big object
    noise_type_set = {1, 2, 3, 5, 8, 10, 11, 12, 13, 15, 18, 19, 22, 23}
    for i in range(len(infer_segLabels)):
        if infer_segLabels[i] == 0 and segLabel[i] != 0 and segLabel[i] != 16:
            segLabel[i] = 0
            count[0] = count[0] + 1
        elif infer_segLabels[i] == 16  and segLabel[i] != 0 and segLabel[i] != 16 and (int(segLabel[i].item()) in noise_type_set):
            segLabel[i] = 16
            count[1] = count[1] + 1
    # print("count(ground/noise) = ", count[0], ",", count[1])

    # Save the PCD file
    da_io.save_fusion_with_seglabel(save_pcd_path, cloud, segLabel)

def findLabelSave(pcd_pth, save_pth):
    pcd_name = os.path.basename(pcd_pth)
    base_name, extension = os.path.splitext(pcd_name)
    label_file_list = glob.glob(os.path.join(args.result_base_dir, base_name+"*pred.npy"))
    if len(label_file_list) == 1:
        segmentation_data = np.load(label_file_list[0])
        # scan = imo_pcd_reader.read_pcd(pcd_pth)
        save_pcd_path = os.path.join(save_pth, base_name+".pcd")
        #imo_pcd_reader.save_pcd(scan, segmentation_data, save_pcd_path)
        # save_pcd_use_noise(pcd_pth, scan, segmentation_data, save_pcd_path)
        save_pcd_use_noise(pcd_pth, segmentation_data, save_pcd_path)
    else:
        print(f"Found {len(label_file_list)} seg label file for {pcd_pth}")

print("start seg2pcd denoise")

save_dir = os.path.abspath(os.path.join(args.result_base_dir, os.pardir))

pc_extension = ".pcd"

occ_frame_pcd_dir = args.data_base_dir
pcd_names = [f for f in os.listdir(occ_frame_pcd_dir) if f.endswith(pc_extension)]
pcd_names.sort()
with Parallel(n_jobs=max(1, int(multiprocessing.cpu_count()/2)), backend='loky') as parallel:
    parallel(delayed(findLabelSave)(os.path.join(occ_frame_pcd_dir, pcd_n), save_dir) 
                for pcd_n in tqdm(pcd_names, total=len(pcd_names), mininterval=1.0))
