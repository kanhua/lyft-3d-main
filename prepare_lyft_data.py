# Prepare ground_truth data
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle
from typing import Tuple
from PIL import Image

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, \
    points_in_box, BoxVisibility

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
    first_train_id, first_train_sample_box = get_train_data_sample_token_and_box(idx, train_objects)

    first_train_sample = level5data.get('sample', first_train_id)

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


def get_train_data_sample_token_and_box(idx, train_objects):
    first_train_id = train_objects.iloc[idx, 0]
    # Make box
    orient_q = Quaternion(axis=[0, 0, 1], angle=float(train_objects.loc[idx, 'yaw']))
    center_pos = train_objects.iloc[idx, 2:5].values
    wlh = train_objects.iloc[idx, 5:8].values
    obj_name = train_objects.iloc[idx, -1]
    first_train_sample_box = Box(center=list(center_pos), size=list(wlh),
                                 orientation=orient_q, name=obj_name)
    return first_train_id, first_train_sample_box


def extract_boxed_clouds(num_entries, point_threshold=1024, while_list_type_str=['car', 'pedestrian', 'bicycle'],
                         save_file=os.path.join(ARTIFACT_PATH, "val_pc.pickle")):
    point_clouds_list = []
    one_hot_vector_list = []

    train_objects = parse_train_csv()

    # get train box information, center and wlh
    for idx in range(train_objects.shape[0]):
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

        # for k in range(masked_ldp_points.shape[1]):
        #    masked_ldp_points[0:3, k] -= box.center

        # Show number of points
        print("number of cloud points: {}".format(masked_ldp_points.shape[1]))

        ##Patch: calibration using KITTI calibration data
        ncpc = calib_point_cloud(masked_ldp_points[0:3, :].T)
        masked_ldp_points[0:3, :] = np.transpose(ncpc)

        rescale_lidar_intensity(masked_ldp_points, 0.2)

        if masked_ldp_points.shape[1] > point_threshold:
            # Store the results
            point_clouds_list.append(masked_ldp_points)
            one_hot_vector_list.append(one_hot_vector)

        if len(point_clouds_list) >= num_entries:
            break

    point_clouds_list = rearrange_point_clouds(point_clouds_list)
    one_hot_vector_list = rearrange_one_hot_vector(one_hot_vector_list)

    # save the file
    with open(save_file, 'wb') as fp:
        save_dict = {"pcl": point_clouds_list,
                     "ohv": one_hot_vector_list}
        pickle.dump(save_dict, fp)

    return point_clouds_list, one_hot_vector_list


def rescale_lidar_intensity(point_cloud: np.ndarray, value: float):
    point_cloud[3, :] = value
    return point_cloud


def calib_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    """
       Calibrate point cloud data using KITTI calib file.
    This is not correct, just for preliminary trying
    :param point_cloud:
    :return: Nx3 array
    """

    from kitti_util import Calibration

    assert point_cloud.shape[1] == 3

    calib_file_path = "/Users/kanhua/Downloads/frustum-pointnets/dataset/KITTI/object/training/calib/000000.txt"
    calib = Calibration(calib_file_path)

    return calib.project_velo_to_rect(point_cloud)


def rearrange_point_clouds(point_clouds_list, num_points=1024):
    num_clouds = len(point_clouds_list)
    output_point_clouds = np.empty((num_clouds, num_points, 4))

    for pc_id, pc in enumerate(point_clouds_list):
        sel_index = np.random.choice(pc.shape[1], num_points)
        assert num_points == sel_index.shape[0]
        output_point_clouds[pc_id, :, :] = np.transpose(pc[:, sel_index])

    return output_point_clouds


def rearrange_one_hot_vector(one_hot_vector_list):
    return np.stack(one_hot_vector_list)


def get_sample_images(sample_token: str, ax=None):
    record = level5data.get("sample", sample_token)

    # Separate RADAR from LIDAR and vision.
    radar_data = {}
    nonradar_data = {}

    for channel, token in record["data"].items():
        sd_record = level5data.get("sample_data", token)
        sensor_modality = sd_record["sensor_modality"]
        if sensor_modality in ["lidar", "camera"]:
            nonradar_data[channel] = token
        else:
            radar_data[channel] = token

    # get projective matrix

    for channel, token in nonradar_data.items():
        sd_record = level5data.get("sample_data", token)
        sensor_modality = sd_record["sensor_modality"]

        if sensor_modality == "camera":
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = level5data.get_sample_data(token, box_vis_level=BoxVisibility.ANY)

            data = Image.open(data_path)

            # Init axes.
            fig, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            for box in boxes:
                c = np.array(LyftDatasetExplorer.get_color(box.name)) / 255.0
                box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

            fig.savefig("./temp_figs/{}.png".format(token), dpi=300)
            plt.close()


def get_pc_in_image_fov(point_cloud_token, camera_type: str):
    """

    :param point_cloud_token:
    :param camera_type: available types: 'CAM_BACK', 'CAM_FRONT_ZOOMED','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK_LEFT','LIDAR_TOP','LIDAR_FRONT_LEFT',
    :param Box:
    :return:
    """

    pc_record = level5data.get("sample_data", point_cloud_token)
    sample_of_pc_record = level5data.get("sample", pc_record['sample_token'])

    cam_token = sample_of_pc_record['data'][camera_type]

    lidar_file_path = level5data.get_sample_data_path(point_cloud_token)
    lpc = LidarPointCloud.from_file(lidar_file_path)

    _, _, img, mask = map_pointcloud_to_image(point_cloud_token, cam_token)

    return lpc.points[:, mask], img

    # Get xmax,xmin,ymax,ymin of the box projected on a image (back, front, etc.)

    # Project point cloud on to the image

    # select the indicies that are within the projected boundary


def map_pointcloud_to_image(pointsensor_token: str, camera_token: str) -> Tuple:
    """Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to
    the image plane.
    The code is adapted from lyft/nuScenes-devkit: lyft_dataset_sdk.lyftdataset.LyftDatasetExplorer.map_pointcloud_to_image()

    Args:
        pointsensor_token: Lidar/radar sample_data token.
        camera_token: Camera sample_data token.

    Returns: (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).

    """

    lyftd = level5data
    cam = lyftd.get("sample_data", camera_token)
    pointsensor = lyftd.get("sample_data", pointsensor_token)
    pcl_path = lyftd.data_path / pointsensor["filename"]
    assert pointsensor["sensor_modality"] == "lidar"
    pc = LidarPointCloud.from_file(pcl_path)

    im = Image.open(str(lyftd.data_path / cam["filename"]))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = lyftd.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))

    # Second step: transform to the global frame.
    poserecord = lyftd.get("ego_pose", pointsensor["ego_pose_token"])
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pc.translate(np.array(poserecord["translation"]))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = lyftd.get("ego_pose", cam["ego_pose_token"])
    pc.translate(-np.array(poserecord["translation"]))
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = lyftd.get("calibrated_sensor", cam["calibrated_sensor_token"])
    pc.translate(-np.array(cs_record["translation"]))
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Retrieve the color from the depth.
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, im, mask
