# Prepare ground_truth data
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, points_in_box

import matplotlib.pyplot as plt

# Load the dataset
# Adjust the dataroot parameter below to point to your local dataset path.
# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps, v1.0.1-train
data_path = '/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles'
ARTIFACT_PATH = "/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/artifacts"

DATA_PATH = '/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/'
level5data_snapshot_file = "level5data.pickle"

if os.path.exists(os.path.join(DATA_PATH, level5data_snapshot_file)):
    with open(os.path.join(DATA_PATH, level5data_snapshot_file), 'rb') as fp:
        level5data = pickle.load(fp)
else:

    level5data = LyftDataset(data_path='/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles',
                             json_path='/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/data/',
                             verbose=True)
    with open(os.path.join(DATA_PATH, level5data_snapshot_file), 'wb') as fp:
        pickle.dump(level5data, fp)


def parse_train_csv():
    train = pd.read_csv(DATA_PATH + 'train.csv')

    object_columns = ['sample_id', 'object_id', 'center_x', 'center_y', 'center_z',
                      'width', 'length', 'height', 'yaw', 'class_name']
    objects = []
    for sample_id, ps in tqdm(train.values[:]):
        object_params = ps.split()
        n_objects = len(object_params)
        for i in range(n_objects // 8):
            x, y, z, w, l, h, yaw, c = tuple(object_params[i * 8: (i + 1) * 8])
            objects.append([sample_id, i, float(x), float(y), float(z), float(w), float(l), float(h), yaw, c])
    train_objects = pd.DataFrame(
        objects,
        columns=object_columns
    )
    return train_objects


def extract_single_box(train_objects, idx) -> Box:
    first_train_id = train_objects.iloc[idx, 0]

    first_train_sample = level5data.get('sample', first_train_id)

    # Make box
    orient_q = Quaternion(axis=[0, 0, 1], angle=float(train_objects.loc[idx, 'yaw']))
    center_pos = train_objects.iloc[idx, 2:5].values
    wlh = train_objects.iloc[idx, 5:8].values
    obj_name = train_objects.iloc[idx, -1]
    first_train_sample_box = Box(center=list(center_pos), size=list(wlh),
                                 orientation=orient_q,name=obj_name)

    sample_data_token = first_train_sample['data']['LIDAR_TOP']

    sd_record = level5data.get("sample_data", sample_data_token)
    cs_record = level5data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = level5data.get("sensor", cs_record["sensor_token"])
    pose_record = level5data.get("ego_pose", sd_record["ego_pose_token"])
    # Move box to ego vehicle coord system
    first_train_sample_box.translate(-np.array(pose_record["translation"]))
    first_train_sample_box.rotate(Quaternion(pose_record["rotation"]).inverse)

    #  Move box to sensor coord system
    first_train_sample_box.translate(-np.array(cs_record["translation"]))
    first_train_sample_box.rotate(Quaternion(cs_record["rotation"]).inverse)

    return first_train_sample_box, sample_data_token


def extract_boxed_clouds(num_entries, while_list_type_str=['car', 'pedestrian', 'bicycle'],
                         save_file=os.path.join(ARTIFACT_PATH, "val_pc.pickle")):
    point_clouds_list = []
    one_hot_vector_list = []

    train_objects = parse_train_csv()

    # get train box information, center and wlh
    for idx in range(num_entries):
        box, sample_data_token = extract_single_box(train_objects, idx=idx)

        print(box.name)
        if box.name not in while_list_type_str:
            continue
        else:
            type_index = while_list_type_str.index(box.name)
            one_hot_vector = np.zeros(3, dtype=np.bool)
            one_hot_vector[type_index] = True

        lidar_file_path = level5data.get_sample_data_path(sample_data_token)
        lpc = LidarPointCloud.from_file(lidar_file_path)

        # get the point cloud associated with this cloud

        # mask out the point clouds
        mask = points_in_box(box, lpc.points[0:3, :])
        masked_ldp_points = lpc.points[:, mask]

        # transform the masked point clouds to frustum coordinates
        # For the time being, just translate it the center coordinates

        for k in range(masked_ldp_points.shape[1]):
            masked_ldp_points[0:3, k] -= box.center

        # Store the results
        point_clouds_list.append(masked_ldp_points)
        one_hot_vector_list.append(one_hot_vector)

    # save the file
    with open(save_file, 'wb') as fp:
        save_dict={"pcl":point_clouds_list,
                   "ohv":one_hot_vector_list}
        pickle.dump(save_dict, fp)

    return point_clouds_list, one_hot_vector_list
