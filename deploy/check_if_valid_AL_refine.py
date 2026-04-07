import argparse
import numpy as np

# import imo_pcd_reader
from da_binds import io as da_io

parser = argparse.ArgumentParser(description='Check if AL segmentation is valid')

parser.add_argument('--seged_pcd_path', type=str, 
                    default='/your/path/to/.../segmented_pcd/*.pcd')

args = parser.parse_args()

def check(seged_pcd_path):
    included_area = np.array([[-0.3, 3], [-0.8, 0.8]])
    excluded_area = np.array([[0.7, 0.95], [-0.85, 0.85]])
    floor_height = 0.5
    ceiling_height = 2
    excluded_label = [0, 16]
    # scan_data = imo_pcd_reader.read_AL_pcd_with_excluded_area_and_included_area(seged_pcd_path, excluded_area, included_area, floor_height, ceiling_height)
    xyzi, seg_label = da_io.read_pcd_xyzis_structured(seged_pcd_path)

    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    # 1. (x, y) is inside included_area
    cond_included = (
        (x >= included_area[0, 0]) & (x <= included_area[0, 1]) &
        (y >= included_area[1, 0]) & (y <= included_area[1, 1])
    )
    # 2. (x, y) is NOT inside excluded_area
    cond_excluded = ~(
        (x >= excluded_area[0, 0]) & (x <= excluded_area[0, 1]) &
        (y >= excluded_area[1, 0]) & (y <= excluded_area[1, 1])
    )
    # 3. z is between floor_height and ceiling_height
    cond_height = (z >= floor_height) & (z <= ceiling_height)
    # 4. seg_label is not in excluded_label
    cond_label = ~np.isin(seg_label, excluded_label)
    # Combined condition
    mask = (cond_included & cond_excluded & cond_height).reshape(-1, 1) & cond_label
    # Filter xyzi and seg_label
    filtered_seg_label = seg_label[mask]

    if len(filtered_seg_label) < 10:
        return True
    else:
        label_count = np.sum(filtered_seg_label == 2)
        proportion = label_count / len(filtered_seg_label)
        if proportion < 0.65:
            return False
        else:
            return True

if __name__ == "__main__":
    result = check(args.seged_pcd_path)
    # print('True' if result else 'False')
    # 返回 0 表示 True，1 表示 False
    exit(0 if result else 1)