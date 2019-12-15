from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle
from typing import Tuple, List
import tensorflow as tf
from PIL import Image

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, \
    points_in_box, BoxVisibility

import matplotlib.pyplot as plt
import warnings

from provider import rotate_pc_along_y

from model_util import g_type2class, g_type_mean_size, g_class2type, NUM_HEADING_BIN


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


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - \
                     (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual


class FrustumGenerator(object):

    def __init__(self, sample_token: str, lyftd: LyftDataset,
                 camera_type=None):

        self.object_of_interest_name = ['car', 'pedestrian', 'cyclist']
        if camera_type is None:
            camera_type = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT',
                           'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
        self.lyftd = lyftd
        self.sample_record = self.lyftd.get("sample", sample_token)
        self.camera_type = camera_type
        self.camera_keys = self._get_camera_keys()
        self.point_cloud_in_sensor_coord, self.ref_lidar_token = self._read_pointcloud(use_multisweep=True)

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
        max_clip_distance = 60

        for cam_key in self.camera_keys:
            camera_token = self.sample_record['data'][cam_key]
            camera_data = self.lyftd.get('sample_data', camera_token)
            point_cloud, lidar_token = self._read_pointcloud(use_multisweep=False)
            # lidar data needs to be reloaded every time
            # Idea: this part can be expanded: different camera image data could use different lidar data,
            # CAM_FRONT_RIGHT can also use LIDAR_FRONT_RIGHT

            image_path, box_list, cam_intrinsic = self.lyftd.get_sample_data(camera_token,
                                                                             box_vis_level=BoxVisibility.ANY,
                                                                             selected_anntokens=None)
            img = Image.open(image_path)

            _, point_cloud_in_camera_coord_2d = transform_pc_to_camera_coord(camera_data,
                                                                             self.lyftd.get('sample_data', lidar_token),
                                                                             point_cloud, self.lyftd)

            for box_in_sensor_coord in box_list:

                # get frustum points

                mask = mask_points(point_cloud_in_camera_coord_2d, 0, img.size[0], ymin=0, ymax=img.size[1])

                distance_mask = (point_cloud.points[2, :] > clip_distance) & (
                        point_cloud.points[2, :] < max_clip_distance)

                mask = np.logical_and(mask, distance_mask)

                projected_corners_8pts = get_box_corners(box_in_sensor_coord, cam_intrinsic,
                                                         frustum_pointnet_convention=True)

                xmin, xmax, ymin, ymax = get_2d_corners_from_projected_box_coordinates(projected_corners_8pts)

                box_mask = mask_points(point_cloud_in_camera_coord_2d, xmin, xmax, ymin, ymax)
                mask = np.logical_and(mask, box_mask)

                point_clouds_in_box = point_cloud.points[:, mask]

                _, seg_label = extract_pc_in_box3d(point_clouds_in_box[0:3, :], box_in_sensor_coord.corners())

                heading_angle = get_box_yaw_angle_in_camera_coords(box_in_sensor_coord)
                frustum_angle = get_frustum_angle(self.lyftd, camera_token, xmax, xmin, ymax, ymin)
                point_clouds_in_box = point_clouds_in_box[0:3, :]
                point_clouds_in_box = np.transpose(point_clouds_in_box)
                box_2d_pts = np.array([xmin, ymin, xmax, ymax])
                box_3d_pts = np.transpose(box_in_sensor_coord.corners())

                #TODO filter out data
                if box_in_sensor_coord.name not in self.object_of_interest_name:
                    continue


                fp = FrusutmPoints(lyftd=self.lyftd, box_in_sensor_coord=box_in_sensor_coord,
                                   point_cloud_in_box=point_clouds_in_box,
                                   box_3d_pts=box_3d_pts, box_2d_pts=box_2d_pts, heading_angle=heading_angle,
                                   frustum_angle=frustum_angle, camera_token=camera_token,
                                   sample_token=self.sample_record['token'], seg_label=seg_label)

                yield fp


class FrusutmPoints(object):
    def __init__(self, lyftd: LyftDataset, box_in_sensor_coord: Box, point_cloud_in_box, box_3d_pts,
                 box_2d_pts, heading_angle, frustum_angle, sample_token, camera_token, seg_label: np.ndarray):
        self.box_in_sensor_coord = box_in_sensor_coord
        self.point_cloud_in_box = point_cloud_in_box
        self.box_3d_pts = box_3d_pts
        self.box_2d_pts = box_2d_pts
        self.heading_angle = heading_angle
        self.frustum_angle = frustum_angle
        self.sample_token = sample_token
        self.camera_token = camera_token
        self.lyftd = lyftd
        self.camera_intrinsic = self._get_camera_intrinsic()
        self.seg_label = seg_label
        self.object_of_interest_name = ['car', 'pedestrian', 'cyclist']

    def _get_center_view_rotate_angle(self):
        return np.pi / 2 + self.frustum_angle

    def _get_rotated_center(self) -> np.array:
        box3d_center = np.copy(self.box_in_sensor_coord.center)

        return rotate_pc_along_y(np.expand_dims(box3d_center, 0),
                                 rot_angle=self._get_center_view_rotate_angle()).squeeze()

    def _get_rotated_point_cloud(self):
        point_cloud = np.copy(self.point_cloud_in_box)

        return rotate_pc_along_y(point_cloud, rot_angle=self._get_center_view_rotate_angle())

    def _get_rotated_box_3d(self):
        r_box_3d_pts = np.copy(self.box_3d_pts)

        return rotate_pc_along_y(r_box_3d_pts,
                                 rot_angle=self._get_center_view_rotate_angle())

    def _get_angle_class_residual(self,rotated_heading_angle):
        angle_class, angle_residual = angle2class(rotated_heading_angle, NUM_HEADING_BIN)

        return angle_class,angle_residual

    def _get_size_class_residual(self):

        #TODO size2class() and settings were copied from size, we therefor use
        # self._get_wlh() instead of self.box_sensor_coord.size
        size_class,size_residual =size2class(self._get_wlh(),self.box_in_sensor_coord.name)
        return size_class,size_residual


    def _get_one_hot_vec(self):
        pass

    def _get_rotated_heading_angle(self):

        return self.heading_angle-self.frustum_angle

    def _get_camera_intrinsic(self) -> np.ndarray:
        sd_record = self.lyftd.get("sample_data", self.camera_token)
        cs_record = self.lyftd.get("calibrated_sensor", sd_record["calibrated_sensor_token"])

        camera_intrinsic = np.array(cs_record['camera_intrinsic'])

        return camera_intrinsic

    def _get_wlh(self):
        w, l, h = self.box_in_sensor_coord.wlh
        size_lwh = np.array([l, w, h])
        return size_lwh

    def _flat_pointclout(self):
        # not support lidar data with intensity yet
        assert self.point_cloud_in_box.shape[1] == 3

        return self.point_cloud_in_box.ravel()

    def to_train_example(self) -> tf.train.Example:

        rotated_heading_angle = self._get_rotated_heading_angle()
        rotated_angle_class,rotated_angle_residual=self._get_angle_class_residual(rotated_heading_angle)

        size_class,size_residual=self._get_size_class_residual()

        feature_dict = {
            'box3d_size': float_list_feature(self._get_wlh()),  # (3,)
            'size_class': int64_feature(size_class),
            'size_residual': float_list_feature(size_residual.ravel()),  # (3,)

            'frustum_point_cloud': float_list_feature(self._flat_pointclout()),  # (N,3)
            'rot_frustum_point_cloud': float_list_feature(self._get_rotated_point_cloud().ravel()),  # (N,3)

            'box_3d': float_list_feature(self.box_3d_pts.ravel()),  # (8,3)
            'rot_box_3d': float_list_feature(self._get_rotated_box_3d().ravel()),  # (8,3)

            'box_2d': float_list_feature(self.box_2d_pts.ravel()),  # (4,)

            'heading_angle': float_feature(self.heading_angle),
            'rot_heading_angle': float_feature(self._get_rotated_heading_angle()),
            'rot_angle_class': int64_feature(rotated_angle_class),
            'rot_angle_residual': float_feature(rotated_angle_residual),

            'frustum_angle': float_feature(self.frustum_angle),
            'sample_token': bytes_feature(self.sample_token.encode('utf8')),
            'type_name': bytes_feature(self.box_in_sensor_coord.name.encode('utf8')),
            'camera_token:': bytes_feature(self.camera_token.encode('utf8')),
            'annotation_token': bytes_feature(self.box_in_sensor_coord.token.encode('utf8')),

            'box_center': float_list_feature(self.box_in_sensor_coord.center.ravel()),  # (3,)
            'rot_box_center': float_list_feature(self._get_rotated_center().ravel()),  # (3,)

        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        return example

    def render_point_cloud_on_image(self, ax):
        projected_pts = view_points(np.transpose(self.point_cloud_in_box),
                                    view=self.camera_intrinsic, normalize=True)
        ax.scatter(projected_pts[0, :], projected_pts[1, :], s=0.1, alpha=0.1)
        # self.lyftd.render_pointcloud_in_image()

    def render_boxes_on_image(self, ax):
        self.box_in_sensor_coord.render(ax, view=self.camera_intrinsic, normalize=True)

    def render_point_cloud_top_view(self, ax, view_matrix=np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]])):
        # self.box_in_sensor_coord.render(ax, view=view_matrix, normalize=False)
        projected_pts = view_points(np.transpose(self.point_cloud_in_box), view=view_matrix, normalize=False)
        ax.scatter(projected_pts[0, :], projected_pts[1, :], s=0.1)
        if projected_pts.shape[1] > 0:
            ax.text(np.mean(projected_pts[0, :]), np.mean(projected_pts[1, :]),
                    "{:.2f}".format(self.frustum_angle * 180 / np.pi))

    def render_image(self, ax):
        image_path = self.lyftd.get_sample_data_path(self.camera_token)
        import skimage
        image_array = skimage.io.imread(image_path)
        ax.imshow(image_array)

    def render_rotated_point_cloud_top_view(self, ax,
                                            view_matrix=np.array([[1, 0, 0],
                                                                  [0, 0, 1], [0, 0, 0]])):
        pc = self._get_rotated_point_cloud()
        projected_pts = view_points(np.transpose(pc), view=view_matrix, normalize=False)
        ax.scatter(projected_pts[0, :], projected_pts[1, :], s=0.1)


def parse_frustum_point_record(tfexample_message: str):
    keys_to_features = {
        "box3d_size": tf.FixedLenFeature((3,), tf.float32),
        "frustum_point_cloud": tf.FixedLenSequenceFeature((3,), tf.float32, allow_missing=True),
        "box_3d": tf.FixedLenFeature((8, 3), tf.float32),
        "box_2d": tf.FixedLenFeature((4,), tf.float32),
        "type_name": tf.FixedLenFeature((), tf.string),
    }

    parsed_example = tf.io.parse_single_example(tfexample_message, keys_to_features)

    return parsed_example


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


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
            for fp in fg.generate_frustums():
                yield fp
        else:
            # reserved for rgb detection data
            pass

        next_sample_token = sample_record['next']
        sample_token = next_sample_token
