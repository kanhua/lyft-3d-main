import pickle
import numpy as np

from provider import class2angle, class2size
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from prepare_lyft_data import get_sensor_to_world_transform_matrix, \
    get_sensor_to_world_transform_matrix_from_sample_data_token, \
    convert_box_to_world_coord,convert_box_to_world_coord_with_sample_data_token
from lyft_dataset_sdk.utils.data_classes import Box, Quaternion


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    npc = np.copy(pc)
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    npc[:, [0, 2]] = np.dot(npc[:, [0, 2]], np.transpose(rotmat))
    return npc


def read_frustum_pointnet_output(pickle_file):
    with open(pickle_file, 'rb') as fp:
        ps_list = pickle.load(fp)
        seg_list = pickle.load(fp)
        segp_list = pickle.load(fp)
        center_list = pickle.load(fp)
        heading_cls_list = pickle.load(fp)
        heading_res_list = pickle.load(fp)
        size_cls_list = pickle(fp)
        size_res_list = pickle(fp)
        rot_angle_list = pickle(fp)
        score_list = pickle(fp)
    return ps_list


def get_heading_angle(heading_cls, heading_res, rot_angle):
    pred_angle_radius = class2angle(heading_cls,
                                    heading_res, NUM_HEADING_BIN) + rot_angle

    return pred_angle_radius


def get_size(size_cls, size_res) -> np.ndarray:
    """
    compute size(l,w,h) from size class and residuals

    :param size_cls:
    :param size_res:
    :return: np.ndarray([l,w,h])
    """

    return class2size(size_cls, size_res)


def get_center_in_sensor_coord(center_coord, rot_angle):
    center_before_rotation = rotate_pc_along_y(np.expand_dims(center_coord, 0), rot_angle=-rot_angle).squeeze()

    return center_before_rotation


def get_center_in_world_coord(center_in_sensor_coord, sample_data_token: str):
    """

    :param center_in_sensor_coord: 3xN array
    :param sample_data_token:
    :return: 3xN array
    """
    mtx = get_sensor_to_world_transform_matrix_from_sample_data_token(sample_data_token)

    center_in_sensor_coord_h = np.concatenate((center_in_sensor_coord, np.ones(1)))

    return np.dot(mtx, center_in_sensor_coord_h).ravel()[0:3]


def make_3d_box(heading_cls, heading_res, rot_angle, size_cls, size_res, center_coord, sample_data_token):
    heading_angle = get_heading_angle(heading_cls, heading_res, rot_angle)
    size = get_size(size_cls, size_res)
    center_sensor_coord = (center_coord, rot_angle)
    get_center_in_world_coord(center_sensor_coord, sample_data_token)

    # Make Box
    # The rationale of doing this: to conform the convention of Box class, the car is originally heading to +x axis,
    # with y(left) and z(top). To make the car heading to the angle it should be in the camera coordinate,
    # we have to rotate it by 90 degree around x axis and [theta] degree around y axis, where [theta] is the heading angle

    l, w, h = size

    first_rot = Quaternion(axis=[1, 0, 0], angle=np.pi / 2)
    second_rot = Quaternion(axis=[0, -1, 0], angle=-heading_angle)
    box_in_sensor_coord = Box(center=center_sensor_coord, size=[w, l, h],
                              orientation=second_rot * first_rot)

    box_in_world_coord=convert_box_to_world_coord(box_in_sensor_coord,sample_data_token)

    assert np.abs(box_in_sensor_coord.orientation.axis[0])<=0.02
    assert np.abs(box_in_sensor_coord.orientation.axis[1])<=0.02

    return box_in_world_coord
