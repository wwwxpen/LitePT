import os
import numpy as np
from numba import cuda, int32, jit
import math
import argparse
import json
from scipy.spatial.transform import Rotation
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
import cv2
import addict
import pdb

# import imo_pcd_reader
from da_binds import io as da_io

parser = argparse.ArgumentParser(description='Point Cloud Alignment')

parser.add_argument('--frame_pcd_dir', type=str, 
                    default='/your/path/to/frame/.../*.pcd')

parser.add_argument('--data_json_dir', type=str, 
                    default='/your/path/to//.../*.json')

parser.add_argument('--save_path', type=str, 
                    default="")

parser.add_argument('--car_plate_num', type=str, 
                    default='6083')

parser.add_argument('--voxelSize', type=float, default=0.1)

parser.add_argument('--is_zy', action="store_true", default=False)

args = parser.parse_args()

def parse_mei_cfgs(calib_info):
    cfgs = addict.Dict()

    cfgs.height = int(calib_info['resolution'].split('*')[1])
    cfgs.width = int(calib_info['resolution'].split('*')[0])
    cfgs.mei_ksi = calib_info["ksi"]
    cfgs.mei_k1, cfgs.mei_k2, cfgs.mei_k3, cfgs.mei_k4 = calib_info["k"][:4]
    if len(calib_info["k"]) == 5:
        cfgs.mei_k5 = calib_info["k"][4]
    cfgs.mei_u0, cfgs.mei_v0 = calib_info["center"]
    cfgs.mei_gama1, cfgs.mei_gama2 = calib_info["gama"]
    cfgs.mei_gama1 = cfgs.mei_gama1 * cfgs.mei_gama2

    return cfgs

def get_camera_calib_params(camera_name, calib_tables):
    rotation = calib_tables[camera_name]['rotation']
    quaternion = np.array([rotation[1], rotation[2], rotation[3], rotation[0]])
    rot_camera_to_ego = Rotation.from_quat(quaternion).as_matrix() # camera to ego
    tvec_camera_to_ego = np.array(calib_tables[camera_name]['translation'])
    trans_camera_to_ego = np.eye(4)
    trans_camera_to_ego[:3, :3] = rot_camera_to_ego
    trans_camera_to_ego[:3, 3] = tvec_camera_to_ego
    trans_ego_to_camera = np.linalg.inv(trans_camera_to_ego)
    distortion_model = None
    if 'camera_model' in calib_tables[camera_name].keys():
        distortion_model = calib_tables[camera_name]['camera_model']
    cam_intrinsic = None
    cam_distortion = None
    if 'camera_intrinsic' in calib_tables[camera_name].keys():
        cam_intrinsic = np.array(calib_tables[camera_name]['camera_intrinsic'])
    if 'distortion' in calib_tables[camera_name].keys():
        cam_distortion = np.array(calib_tables[camera_name]['distortion'])
    taylor_coefficient = None
    distortion_center = None
    stretch_matrix = None
    mei_cfgs = None
    if distortion_model == 'ocam':
        taylor_coefficient = np.array(calib_tables[camera_name]['distortion'])
        distortion_center = np.array(calib_tables[camera_name]['center'])
        stretch_matrix = np.array(calib_tables[camera_name]['affine'])
    if distortion_model == 'fisheye':
        taylor_coefficient = np.array(calib_tables[camera_name]['distortion'])
    if distortion_model == 'mei':
        mei_cfgs = parse_mei_cfgs(calib_tables[camera_name])
    return cam_intrinsic, cam_distortion, distortion_model, trans_ego_to_camera, taylor_coefficient, distortion_center, stretch_matrix, mei_cfgs

def ocam_world2cam(world_points, extrinsics, taylor_coefficient, distortion_center, affine, img_size):
    if taylor_coefficient is None or distortion_center is None:
        raise ValueError("Fisheye parameters are empty. You first need to specify or load camera's parameters.")

    # Transform points from world coordinates to camera coordinates
    if extrinsics is not None:
        rotation_matrix = extrinsics[:3, :3]
        translation_vector = extrinsics[:3, 3]
        cam_points = (rotation_matrix.dot(world_points.T) + translation_vector.reshape(3, 1)).T

    norm = np.sqrt(np.sum(cam_points[:, :2]**2, axis=1))

    # Compute theta (angle) for each point
    theta = np.arctan2(-cam_points[:, 2], norm)

    # Compute rho using the polynomial coefficients (apply polynomial expansion)
    theta_powers = np.vstack([theta**i for i in range(len(taylor_coefficient))]).T
    rho = np.sum(theta_powers * np.array(taylor_coefficient), axis=1)

    # Inverse normalization factor
    inv_norm = 1.0 / norm

    # Project to 2D using vectorized operations
    xn = np.column_stack([cam_points[:, 0] * inv_norm * rho, cam_points[:, 1] * inv_norm * rho])

    # Apply affine transformation and principal point
    u = xn[:, 0] * affine[0] + xn[:, 1] * affine[1] + distortion_center[0]
    v = xn[:, 0] * affine[2] + xn[:, 1] + distortion_center[1]

    mask = (cam_points[:, 2] > 0) & (u >= 0) & (u < img_size[0]) & (v >= 0) & (v < img_size[1])

    return mask, u[mask], v[mask]

def mei_world2cam(world_points, extrinsics, mei_intrinsic):
    # Transform points from world coordinates to camera coordinates
    if extrinsics is not None:
        rotation_matrix = extrinsics[:3, :3]
        translation_vector = extrinsics[:3, 3]
        cam_pts = (rotation_matrix.dot(world_points.T) + translation_vector.reshape(3, 1)).T

    ksi = mei_intrinsic.mei_ksi
    k1 = mei_intrinsic.mei_k1
    k2 = mei_intrinsic.mei_k2
    k3 = mei_intrinsic.mei_k3
    k4 = mei_intrinsic.mei_k4
    k5 = mei_intrinsic.get('mei_k5')
    gama1 = mei_intrinsic.mei_gama1
    gama2 = mei_intrinsic.mei_gama2
    u0 = mei_intrinsic.mei_u0
    v0 = mei_intrinsic.mei_v0
    width = mei_intrinsic.width
    height = mei_intrinsic.height

    Rc = np.sqrt(np.sum(np.square(cam_pts[:, :3]), axis=1, keepdims=True))

    cam_pts = cam_pts[:, :3] / Rc
    Xmu = cam_pts[:, :1] / (cam_pts[:, 2:3] + ksi)
    Ymu = cam_pts[:, 1:2] / (cam_pts[:, 2:3] + ksi)

    rho = np.square(Xmu) + np.square(Ymu)
    if k5 is None:
        temp = 1 + k1 * rho + k2 * np.square(rho) + np.power(rho, 3)
    else:
        temp = 1 + k1 * rho + k2 * np.square(rho) + k5 * np.power(rho, 3)
    Xmd = Xmu * temp + 2 * k3 * Xmu * Ymu + k4 * (rho + 2 * np.square(Xmu))
    Ymd = Ymu * temp + 2 * k4 * Xmu * Ymu + k3 * (rho + 2 * np.square(Ymu))

    XmdPoints = np.hstack((np.hstack((Xmd, Ymd)), np.ones(Xmd.shape)))

    K = np.array([[gama1, 0, u0], [0, gama2, v0], [0, 0, 1]], dtype=np.float64)
    img_points = (K @ XmdPoints.T)

    img_points = (img_points[:2, :] / img_points[2, :]).round().astype(int)

    mask = ((cam_pts[:, 2] > 0) &\
            (img_points[0, :] >= 0) &\
            (img_points[0, :] < width) &\
            (img_points[1, :] >= 0) &\
            (img_points[1, :] < height))


    return mask, img_points[0, mask], img_points[1, mask]

def fisheye_visible_mask(cam_intrinsic, extrinsics, distcoeff, pcd_coords, img_size):
    rotation_matrix = extrinsics[:3, :3]
    translation_vector = extrinsics[:3, 3]

    # transform to camera coordinate
    cam_points = rotation_matrix.dot(pcd_coords.T) + translation_vector.reshape(3, 1)

    # normalize and project to image plane
    epsilon = 1e-6
    x_normalized = cam_points[0, :] / (cam_points[2, :] + epsilon)
    y_normalized = cam_points[1, :] / (cam_points[2, :] + epsilon)
    undistorted_points = np.vstack([x_normalized, y_normalized]).T
    undistorted_points = np.array(undistorted_points, dtype=np.float32)
    undistorted_points = undistorted_points.reshape(-1, 1, 2)
    distorted_points = cv2.fisheye.distortPoints(undistorted_points, cam_intrinsic, distcoeff)
    distorted_points = distorted_points.reshape(-1, 2)

    mask = (
        (cam_points[2, :] > 0) &
        (distorted_points[:, 0] >= 0) & (distorted_points[:, 0] < img_size[0]) &
        (distorted_points[:, 1] >= 0) & (distorted_points[:, 1] < img_size[1])
    )
    
    return mask, distorted_points[:, 0][mask], distorted_points[:, 1][mask]

def project_devself_points(extrinsic_mat, intrinsic_mat, distort_mat, pcd_coords, img_size):

    # 将外参矩阵转换为 numpy 数组（假设它是一个 4x4 的矩阵）
    extrinsic = np.array(extrinsic_mat).reshape(4, 4)
    
    # 将内参矩阵转换为 numpy 数组
    intrinsic = np.array(intrinsic_mat).reshape(3, 3)
    
    # 畸变系数
    distcoeff = np.array(distort_mat).reshape(1, -1)

    pts_3d = []

    # 提取 rvec 和 tvec
    rotationMatrix = extrinsic[:3, :3]  # R
    tvec = extrinsic[:3, 3]  # T
    rvec, _ = cv2.Rodrigues(rotationMatrix)
        
    # for point in cloud.points:
    for point in pcd_coords:
        
        tmpxC, tmpyC, tmpzC = point[0], point[1], point[2]
        
        pts_3d.append(np.array([tmpxC, tmpyC, tmpzC])) 

    pts_3d = np.array(pts_3d).reshape(-1, 3)
    distorted_points, _ = cv2.projectPoints(pts_3d, rvec, tvec, intrinsic, distcoeff)
    distorted_points = distorted_points.reshape(-1, 2)

    cam_points = rotationMatrix.dot(pcd_coords.T) + tvec.reshape(3, 1)

    mask = (
        (cam_points[2, :] > 3) & 
        (distorted_points[:, 0] >= 0) & (distorted_points[:, 0] < img_size[0]) &
        (distorted_points[:, 1] >= 0) & (distorted_points[:, 1] < img_size[1])
    )
    
    return mask, distorted_points[:, 0][mask], distorted_points[:, 1][mask]

def lidar2Camera(point, rot_quaternion, translation):
    aug_point = np.append(point, 1)
    quaternion = np.array([[rot_quaternion[1], rot_quaternion[2], rot_quaternion[3], rot_quaternion[0]]])
    translation = np.array(translation)
    rotation = Rotation.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation
    cam_point = np.linalg.inv(homogeneous_matrix) @ aug_point
    return cam_point[:3]

def camera2Px(point, intrinsic):
    intrinsicMatrix = np.array(intrinsic)
    tmp_pt = intrinsicMatrix @ point
    if tmp_pt[2] == 0:
        return np.array([tmp_pt[0], tmp_pt[1]])
    else:
        return np.array([tmp_pt[0]/tmp_pt[2], tmp_pt[1]/tmp_pt[2]])

# Function to determine grid and block sizes
def get_sizes(N, max_block_size=1024):
    # Determine the optimal block size based on GPU capabilities
    device = cuda.get_current_device()
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    if max_block_size > max_threads_per_block:
        max_block_size = max_threads_per_block
    
    # Find the largest power of 2 block size that divides N
    for block_size in range(max_block_size, 0, -1):
        if N % block_size == 0:
            break
    
    # Calculate grid size
    grid_size = (N + block_size - 1) // block_size
    
    return grid_size, block_size

excluded_area = np.array([[0, 0], [0, 0]])
ceiling_height = 10
label_area_range = 200
selected_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27]
voxelSize = args.voxelSize
ten_v_cam_names = ["tvRear", "tvLeft", "tvFront", "tvRight", "svRightRear", "svLeftRear", "svRightFront", "svLeftFront", "rear", "front"]
zy_cam_names = ["tvRear", "tvLeft", "tvFront", "tvRight", "svRightRear", "svLeftRear", "svRightFront", "svLeftFront", "rear", "front", "frontTele"]
four_v_cam_names = ["tvRear", "tvLeft", "tvFront", "tvRight"]
cam_img_size = [
    [1920, 1536],
    [1920, 1536],
    [1920, 1536],
    [1920, 1536],
    [1920, 1080],
    [1920, 1080],
    [1920, 1080],
    [1920, 1080],
    [1920, 1080],
    [3840, 2160],
    [3840, 2160]
]

cam_names = []
if args.is_zy:
    cam_names = zy_cam_names
else:
    if args.car_plate_num == "6083":
        cam_names = ten_v_cam_names
    else:
        cam_names = four_v_cam_names

if args.car_plate_num == "0909":
    cam_img_size[0], cam_img_size[1], cam_img_size[2], cam_img_size[3] = [1920, 1280], [1920, 1280], [1920, 1280], [1920, 1280]

transformData_file = args.data_json_dir
try:
    with open(transformData_file, 'r') as file:
        if args.is_zy:
            calib = json.load(file)
            transformData = {}
            need_modify_camnames = False
            for cam_name in cam_names:
                if cam_name not in calib['default_cameramodel'].keys():
                    need_modify_camnames = True
                    break
            if need_modify_camnames:
                cam_names = four_v_cam_names 
            for cam_name in cam_names:
                cam_model = calib['default_cameramodel'][cam_name]
                transformData[cam_name] = calib[cam_name][cam_model]
                transformData[cam_name]['camera_model'] = cam_model
        else:
            transformData = json.load(file)
except FileNotFoundError:
    print(f"File not found: {transformData_file}")

print(f"start {args.frame_pcd_dir}")
# pcd_data = imo_pcd_reader.read_AL_pcd_selected_label(args.frame_pcd_dir, excluded_area, ceiling_height, label_area_range, selected_labels)
xyzi, seg_label = da_io.read_pcd_xyzis_structured(args.frame_pcd_dir)

x = xyzi[:, 0]
y = xyzi[:, 1]
z = xyzi[:, 2]
# 1. (x, y) is inside included_area
cond_included = (
    (x >= -label_area_range) & (x <= label_area_range) &
    (y >= -label_area_range) & (y <= label_area_range))
# 2. (x, y) is NOT inside excluded_area
cond_excluded = ~(
    (x >= excluded_area[0, 0]) & (x <= excluded_area[0, 1]) &
    (y >= excluded_area[1, 0]) & (y <= excluded_area[1, 1]))
# 3. z <= ceiling_height
cond_height = z <= ceiling_height
# 4. seg_label is in selected_labels
cond_label = np.isin(seg_label, selected_labels)
# Combined condition
mask = cond_included.reshape(-1) & cond_excluded.reshape(-1) & cond_height.reshape(-1) & cond_label.reshape(-1)
# Filter xyzi and seg_label
filtered_xyzi = xyzi[mask]
filtered_seg_label = seg_label[mask]

# cam_vis_arr = np.zeros((pcd_data.shape[0], len(cam_names)), dtype=np.int32)
cam_vis_arr = np.zeros((filtered_xyzi.shape[0], len(cam_names)), dtype=np.int32)
# pcd_coords = pcd_data[:, :3]
pcd_coords = filtered_xyzi[:, :3]
min_coords = np.min(pcd_coords, axis=0)
max_coords = np.max(pcd_coords, axis=0)
grid_size = np.ceil( (max_coords - min_coords) / voxelSize )

grid_index_array_idx_map = {}
row_indices = []

for coord in pcd_coords:
    grid_coords = np.floor( (coord - min_coords) / voxelSize )
    grid_index = (grid_coords[0], grid_coords[1], grid_coords[2])
    if grid_index not in grid_index_array_idx_map:
        grid_index_array_idx_map[grid_index] = 1
        row_index = grid_coords[0] * grid_size[1] * grid_size[2] + grid_coords[1] * grid_size[2] + grid_coords[2]
        row_indices.append(row_index)

row_indices.sort()

# Convert to device arrays
d_row_indices = cuda.to_device(row_indices)

# Number of non-zero elements (non-void voxels)
num_non_voids = len(d_row_indices)

@cuda.jit
def ray_trace_kernel(start_points, end_point, min_coords, voxelSize, hit_results, row_indices, grid_size, vert_idx):
    idx = cuda.grid(1)
    
    if idx < start_points.shape[0]:
        if vert_idx > 0 and hit_results[idx] == 0:
            return
        ox, oy, oz = int((start_points[idx, 0] - min_coords[0])/voxelSize), int((start_points[idx, 1] - min_coords[1])/voxelSize), int((start_points[idx, 2] - min_coords[2])/voxelSize)
        ray_vec_x, ray_vec_y, ray_vec_z = start_points[idx, 0] - end_point[0], start_points[idx, 1] - end_point[1], start_points[idx, 2] - end_point[2]
        o_length = math.sqrt(ray_vec_x**2 + ray_vec_y**2 + ray_vec_z**2)
        step = voxelSize
        walk = step
        c_length = 1 - (walk/o_length)
        curr_pos_x, curr_pos_y, curr_pos_z = end_point[0] + c_length*ray_vec_x, end_point[1] + c_length*ray_vec_y, end_point[2] + c_length*ray_vec_z
        pos_x, pos_y, pos_z = int((curr_pos_x - min_coords[0])/voxelSize), int((curr_pos_y - min_coords[1])/voxelSize), int((curr_pos_z - min_coords[2])/voxelSize)
        while c_length > 0 and 0 <= pos_x < grid_size[0] and 0 <= pos_y < grid_size[1] and 0 <= pos_z < grid_size[2]:
            if (pos_x, pos_y, pos_z) != (ox, oy, oz):
                voxel_index = pos_x * grid_size[1] * grid_size[2] + pos_y * grid_size[2] + pos_z
                # Binary search to find if the voxel is in the COO data
                left, right = 0, num_non_voids - 1
                while left <= right:
                    mid = (left + right) // 2
                    if row_indices[mid] == voxel_index:
                        hit_results[idx] = 1
                        return
                    elif row_indices[mid] < voxel_index:
                        left = mid + 1
                    else:
                        right = mid - 1
            
            walk = walk + step
            c_length = 1 - (walk/o_length)
            curr_pos_x, curr_pos_y, curr_pos_z = end_point[0] + c_length*ray_vec_x, end_point[1] + c_length*ray_vec_y, end_point[2] + c_length*ray_vec_z
            pos_x, pos_y, pos_z = int((curr_pos_x - min_coords[0])/voxelSize), int((curr_pos_y - min_coords[1])/voxelSize), int((curr_pos_z - min_coords[2])/voxelSize)
        
        hit_results[idx] = hit_results[idx] and 0

@jit(nopython=True)
def add_vertices(arr, vertices):
    for i in range(len(arr)):
        arr[i, :] += vertices

for cam_idx, cam_n in enumerate(cam_names):
    # Define start points and directions
    end_point = np.array(transformData[cam_n]["translation"], dtype=np.float32)
    d_end_point = cuda.to_device(end_point)
    d_min_coords = cuda.to_device(min_coords)

    vertices = [np.array([0, 0, 0]),
                np.array([-voxelSize/2, -voxelSize/2, -voxelSize/2]), 
                np.array([voxelSize/2, -voxelSize/2, -voxelSize/2]),
                np.array([voxelSize/2, voxelSize/2, -voxelSize/2]),
                np.array([-voxelSize/2, voxelSize/2, -voxelSize/2]),
                np.array([-voxelSize/2, -voxelSize/2, voxelSize/2]), 
                np.array([voxelSize/2, -voxelSize/2, voxelSize/2]),
                np.array([voxelSize/2, voxelSize/2, voxelSize/2]),
                np.array([-voxelSize/2, voxelSize/2, voxelSize/2])]

    hit_results = np.zeros((pcd_coords.shape[0]), dtype=np.int32)
    d_hit_results = cuda.to_device(hit_results)

    for vert_idx, vertice in enumerate(vertices):
        contiguous_pcd_coords = np.ascontiguousarray(pcd_coords)
        add_vertices(contiguous_pcd_coords, vertice)
        d_start_points = cuda.to_device(contiguous_pcd_coords)

        # Define block and grid sizes
        grid_size_cuda, block_size = get_sizes(pcd_coords.shape[0])

        # Launch the kernel
        ray_trace_kernel[grid_size_cuda, block_size](d_start_points, d_end_point, d_min_coords, voxelSize, d_hit_results, d_row_indices, grid_size, vert_idx)

    # Copy results back to host
    hit_results = d_hit_results.copy_to_host()

    hit_results = 1 - hit_results

    hit_results_mask = None
    if args.is_zy:
        cam_intrinsic, cam_distortion, distortion_model, trans_ego_to_camera, taylor_coefficient, \
        distortion_center, stretch_matrix, mei_intrinsic = get_camera_calib_params(cam_n, transformData)
        if transformData[cam_n]["camera_model"] == "ocam":
            hit_results_mask, u, v = ocam_world2cam(pcd_coords, trans_ego_to_camera, taylor_coefficient, distortion_center, stretch_matrix, cam_img_size[cam_idx])
        elif transformData[cam_n]["camera_model"] == "fisheye":
            hit_results_mask, u, v = fisheye_visible_mask(cam_intrinsic, trans_ego_to_camera, taylor_coefficient, pcd_coords, cam_img_size[cam_idx])
        elif transformData[cam_n]["camera_model"] == "pinhole":
            hit_results_mask, u, v = project_devself_points(trans_ego_to_camera, cam_intrinsic, cam_distortion, pcd_coords, cam_img_size[cam_idx])
        elif transformData[cam_n]["camera_model"] == "mei":
            hit_results_mask, u, v = mei_world2cam(pcd_coords, trans_ego_to_camera, mei_intrinsic)
    else:
        def process_point(coord):
            pointInCamera = lidar2Camera(coord, transformData[cam_n]["rotation"], transformData[cam_n]["translation"])
            pointInPx = camera2Px(pointInCamera, transformData[cam_n]["camera_intrinsic"])
            if pointInCamera[2] > 0 and pointInPx[0] > 0 and pointInPx[0] < cam_img_size[cam_idx][0] and pointInPx[1] > 0 and pointInPx[1] < cam_img_size[cam_idx][1]:
                return 1
            else:
                return 0
        hit_results_mask = Parallel(n_jobs=min(8, int(multiprocessing.cpu_count()/2.0)))(delayed(process_point)(coord) for coord in pcd_coords)
    hit_results = hit_results & np.array(hit_results_mask)

    cam_vis_arr[:, cam_idx] = hit_results

pcd_name = os.path.basename(args.frame_pcd_dir)
base_name, extension = os.path.splitext(pcd_name)
save_pth = args.save_path
if not os.path.exists(save_pth): os.makedirs(save_pth)
save_pcd_path = os.path.join(save_pth, base_name+"_occlusion"+".pcd")
print(f"Saving... {save_pcd_path}")
# imo_pcd_reader.save_occlusion_pcd(pcd_data[:, :4], pcd_data[:, 4], cam_vis_arr, save_pcd_path)
da_io.save_pcd_occlusion(filtered_xyzi, filtered_seg_label, cam_vis_arr, save_pcd_path)