from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle
from typing import Tuple, List
from PIL import Image

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, \
    points_in_box, BoxVisibility

import matplotlib.pyplot as plt
import warnings


def transform_pc_to_camera_coord(cam: dict, pointsensor: dict, point_cloud_3d: LidarPointCloud, lyftd: LyftDataset):
    warnings.warn("The point cloud is transformed to camera coordinates in place", UserWarning)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = lyftd.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    point_cloud_3d.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    point_cloud_3d.translate(np.array(cs_record["translation"]))
    # Second step: transform to the global frame.
    poserecord = lyftd.get("ego_pose", pointsensor["ego_pose_token"])
    point_cloud_3d.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    point_cloud_3d.translate(np.array(poserecord["translation"]))
    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = lyftd.get("ego_pose", cam["ego_pose_token"])
    point_cloud_3d.translate(-np.array(poserecord["translation"]))
    point_cloud_3d.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)
    # Fourth step: transform into the camera.
    cs_record = lyftd.get("calibrated_sensor", cam["calibrated_sensor_token"])
    point_cloud_3d.translate(-np.array(cs_record["translation"]))
    point_cloud_3d.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    point_cloud_2d = view_points(point_cloud_3d.points[:3, :],
                                 np.array(cs_record["camera_intrinsic"]), normalize=True)

    return point_cloud_3d, point_cloud_2d


class FrustumGenerator(object):

    def __init__(self, sample_token: str, lyftd: LyftDataset,
                 camera_type=None):
        if camera_type is None:
            camera_type = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT',
                           'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
        self.lyftd = lyftd
        self.sample_record = self.lyftd.get("sample", sample_token)
        self.camera_type = camera_type
        self.camera_keys = self._get_camera_keys()
        self.point_cloud_in_sensor_coord, self.ref_lidar_token = self._read_pointcloud(use_multisweep=False)

    def _get_camera_keys(self):
        cams = [key for key in self.sample_record["data"].keys() if "CAM" in key]
        cams = [cam for cam in cams if cam in self.camera_type]
        return cams

    def _read_pointcloud(self, use_multisweep=False):

        lidar_data_token = self.sample_record['data']['LIDAR_TOP']
        lidar_data_record = self.lyftd.get("sample_data", lidar_data_token)

        pcl_path = self.lyftd.get_sample_data_path(lidar_data_token)

        if use_multisweep:
            pc, _ = LidarPointCloud.from_file_multisweep(self.lyftd, self.sample_record, chan='LIDAR_TOP',
                                                         ref_chan='LIDAR_TOP', num_sweeps=5)
        else:
            pc = LidarPointCloud.from_file(pcl_path)

        return pc, lidar_data_token

    def generate_frustums(self):
        clip_distance = 2.0

        for cam_key in self.camera_keys:
            camera_token = self.sample_record['data'][cam_key]
            camera_data = self.lyftd.get('sample_data', camera_token)
            point_cloud, lidar_token = self._read_pointcloud(use_multisweep=False)
            # lidar data needs to be reloaded every time
            # Idea: this part can be expanded: different camera image data could use different lidar data,
            # CAM_FRONT_RIGHT can also use LIDAR_FRONT_RIGHT

            image_path, box_list, cam_intrinsic = self.lyftd.get_sample_data(camera_token,
                                                                             box_vis_level=BoxVisibility.ALL,
                                                                             selected_anntokens=None)
            img = Image.open(image_path)

            _, point_cloud_in_camera_coord_2d = transform_pc_to_camera_coord(camera_data,
                                                                             self.lyftd.get('sample_data', lidar_token),
                                                                             point_cloud, self.lyftd)

            for box_in_sensor_coord in box_list:
                # get frustum points

                mask = mask_points(point_cloud_in_camera_coord_2d, 0, img.size[0], ymin=0, ymax=img.size[1])

                distance_mask = (point_cloud.points[2, :] > clip_distance)

                mask = np.logical_and(mask, distance_mask)

                projected_corners_8pts = get_box_corners(box_in_sensor_coord, cam_intrinsic,
                                                         frustum_pointnet_convention=True)

                xmin, xmax, ymin, ymax = get_2d_corners_from_projected_box_coordinates(projected_corners_8pts)

                box_mask = mask_points(point_cloud_in_camera_coord_2d, xmin, xmax, ymin, ymax)
                mask = np.logical_and(mask, box_mask)

                point_clouds_in_box = point_cloud.points[:, mask]

                _, label = extract_pc_in_box3d(point_clouds_in_box[0:3, :], box_in_sensor_coord.corners())

                heading_angle = get_box_yaw_angle_in_camera_coords(box_in_sensor_coord)
                frustum_angle = get_frustum_angle(self.lyftd, camera_token, xmax, xmin, ymax, ymin)
                point_clouds_in_box = point_clouds_in_box[0:3, :]
                point_clouds_in_box = np.transpose(point_clouds_in_box)
                box_2d_pts = np.array([xmin, ymin, xmax, ymax])
                w, l, h = box_in_sensor_coord.wlh
                size_lwh = np.array([l, w, h])
                box_3d_pts = np.transpose(box_in_sensor_coord.corners())


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(input_pc, input_box3d):
    """
    The code and in_hull functions are copied from frustum-point on
    https://github.com/charlesq34/frustum-pointnets

    :param point_cloud_3d: 3XN array
    :param box3d: 3x8 array
    :return:
    """

    assert input_box3d.shape == (3, 8)
    assert input_pc.shape[0] == 3
    pc = np.transpose(input_pc)
    box3d = np.transpose(input_box3d)

    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def get_2d_corners_from_projected_box_coordinates(projected_corners: np.ndarray):
    assert projected_corners.shape[0] == 3

    xmin = projected_corners[0, :].min()
    xmax = projected_corners[0, :].max()
    ymin = projected_corners[1, :].min()
    ymax = projected_corners[1, :].max()

    return xmin, xmax, ymin, ymax


def get_box_corners(transformed_box: Box,
                    cam_intrinsic_mtx: np.array,
                    frustum_pointnet_convention=True):
    box_corners_on_cam_coord = transformed_box.corners()

    # Regarrange to conform Frustum-pointnet's convention

    if frustum_pointnet_convention:
        rearranged_idx = [0, 3, 7, 4, 1, 2, 6, 5]
        box_corners_on_cam_coord = box_corners_on_cam_coord[:, rearranged_idx]

        assert np.allclose((box_corners_on_cam_coord[:, 0] + box_corners_on_cam_coord[:, 6]) / 2,
                           np.array(transformed_box.center))

    # For perspective transformation, the normalization should set to be True
    box_corners_on_image = view_points(box_corners_on_cam_coord, view=cam_intrinsic_mtx, normalize=True)

    return box_corners_on_image


def mask_points(points: np.ndarray, xmin,
                xmax, ymin, ymax, depth_min=0, buffer_pixel=1) -> np.ndarray:
    """
    Mask out points outside xmax,xmin,ymin,ymax


    :param points:
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :param depth_min:
    :param buffer_pixel:
    :return: index array
    """
    depths = points[2, :]

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > depth_min)
    mask = np.logical_and(mask, points[0, :] > xmin + buffer_pixel)
    mask = np.logical_and(mask, points[0, :] < xmax - buffer_pixel)
    mask = np.logical_and(mask, points[1, :] > ymin + buffer_pixel)
    mask = np.logical_and(mask, points[1, :] < ymax - buffer_pixel)

    return mask


def get_box_yaw_angle_in_camera_coords(box: Box):
    """
    Calculate the heading angle, using the convention in KITTI labels.

    :param box: bouding box
    :return:
    """

    box_corners = box.corners()
    v = box_corners[:, 0] - box_corners[:, 4]
    heading_angle = np.arctan2(-v[2], v[0])
    return heading_angle


def transform_image_to_cam_coordinate(image_array_p: np.array, camera_token: str, lyftd: LyftDataset):
    sd_record = lyftd.get("sample_data", camera_token)
    cs_record = lyftd.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = lyftd.get("sensor", cs_record["sensor_token"])
    pose_record = lyftd.get("ego_pose", sd_record["ego_pose_token"])

    # inverse the viewpoint transformation
    def normalization(input_array):
        input_array[0:2, :] = input_array[0:2, :] * input_array[2:3, :].repeat(2, 0).reshape(2, input_array.shape[1])
        return input_array

    image_array = normalization(np.copy(image_array_p))
    image_array = np.concatenate((image_array.ravel(), np.array([1])))
    image_array = image_array.reshape(4, 1)

    cam_intrinsic_mtx = np.array(cs_record["camera_intrinsic"])
    view = cam_intrinsic_mtx
    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view
    image_in_cam_coord = np.dot(np.linalg.inv(viewpad), image_array)

    return image_in_cam_coord[0:3, :]


def get_frustum_angle(lyftd: LyftDataset, cam_token, xmax, xmin, ymax, ymin):
    random_depth = 20
    image_center = np.array([[(xmax + xmin) / 2, (ymax + ymin) / 2, random_depth]]).T
    image_center_in_cam_coord = transform_image_to_cam_coordinate(image_center, cam_token, lyftd)
    assert image_center_in_cam_coord.shape[1] == 1
    frustum_angle = -np.arctan2(image_center_in_cam_coord[2, 0], image_center_in_cam_coord[0, 0])
    return frustum_angle


def get_all_boxes_in_single_scene(scene_number, from_rgb_detection, ldf: LyftDataset):
    start_sample_token = ldf.scene[scene_number]['first_sample_token']
    sample_token = start_sample_token
    counter = 0
    while sample_token != "":
        print(counter)
        counter += 1
        sample_record = ldf.get('sample', sample_token)
        if not from_rgb_detection:
            fg = FrustumGenerator(sample_token, ldf)
            fg.generate_frustums()


        else:
            # reserved for rgb detection data
            pass

        next_sample_token = sample_record['next']
        sample_token = next_sample_token