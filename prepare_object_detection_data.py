from typing import List

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, \
    points_in_box, BoxVisibility
from prepare_lyft_data import level5data
from utils import dataset_util
import numpy as np

object_classes = ['car', 'pedestrian', 'animal', 'other_vehicle', 'bus', 'motorcycle', 'truck', 'emergency_vehicle',
                  'bicycle']

object_idx_dict = {'car': 0, 'pedestrian': 1, 'animal': 2, 'other_vehicle': 3,
                   'bus': 4, 'motorcycle': 5, 'truck': 6, 'emergency_vehicle': 7,
                   'bicycle': 8}


def check_object_class_match():
    for idx, cat in enumerate(level5data.category):
        print(cat)
        if object_classes[idx] != cat['name']:
            print("{} does not match {}".format(object_classes[idx], cat['name']))


def get_2d_corners_from_projected_box_coordinates(projected_corners: np.ndarray):
    assert projected_corners.shape[0] == 3

    xmin = projected_corners[0, :].min()
    xmax = projected_corners[0, :].max()
    ymin = projected_corners[1, :].min()
    ymax = projected_corners[1, :].max()

    return xmin, xmax, ymin, ymax


def select_annotation_boxes(sample_token, lyftd: LyftDataset, box_vis_level: BoxVisibility = BoxVisibility.ALL,
                            camera_type=['CAM_FRONT', 'CAM_BACK']) -> (str, str, Box):
    """
    Select annotations that is a camera image defined by box_vis_level


    :param sample_token:
    :param box_vis_level:BoxVisbility.ALL or BoxVisibility.ANY
    :param camera_type: a list of camera that the token should be selected from
    :return: yield (sample_token,cam_token, Box)
    """
    sample_record = lyftd.get('sample', sample_token)

    cams = [key for key in sample_record["data"].keys() if "CAM" in key]
    cams = [cam for cam in cams if cam in camera_type]
    for cam in cams:

        # This step selects all the annotations in a camera image that matches box_vis_level
        cam_token = sample_record["data"][cam]
        image_filepath, boxes_in_sensor_coord, cam_intrinsic = lyftd.get_sample_data(
            cam_token, box_vis_level=box_vis_level, selected_anntokens=sample_record['anns']
        )

        sd_record = lyftd.get('sample_data', cam_token)
        img_width = sd_record['width']  # typically 1920
        img_height = sd_record['height']  # typically 1080

        CORNER_NUMBER = 4
        corner_list = np.empty(np.shape(len(boxes_in_sensor_coord), CORNER_NUMBER))
        for idx, box_in_sensor_coord in enumerate(boxes_in_sensor_coord):
            # For perspective transformation, the normalization should set to be True
            box_corners_on_image = view_points(box_in_sensor_coord.corners(), view=cam_intrinsic, normalize=True)

            corners_2d = get_2d_corners_from_projected_box_coordinates(box_corners_on_image)
            corner_list[idx, :] = corners_2d

        yield image_filepath, corner_list, box_in_sensor_coord


def create_tf_feature(image_file_path: str,
                      camera_token: str,
                      corner_list: np.ndarray,
                      image_width: int, image_heigth: int, boxes: List[Box]):

    box_feature_list = [(box.name, box.token, object_idx_dict[box.name]) for box in boxes]
    box_feature_list =map(list,zip(*box_feature_list))


    feature_dict = {
        'image/height': dataset_util.int64_feature(image_heigth),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(
            image_file_path.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            camera_token.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(corner_list[:, 0]),
        'image/object/bbox/xmax': dataset_util.float_list_feature(corner_list[:, 1]),
        'image/object/bbox/ymin': dataset_util.float_list_feature(corner_list[:, 2]),
        'image/object/bbox/ymax': dataset_util.float_list_feature(corner_list[:, 3]),
        'image/object/class/text': dataset_util.bytes_list_feature(box_feature_list[0]),
        'image/object/class/label': dataset_util.int64_list_feature(box_feature_list[2]),
        'image/object/class/anns_id':dataset_util.bytes_list_feature(box_feature_list[1])
    }

    return feature_dict


if __name__ == "__main__":
    from prepare_lyft_data import get_paths
    import pandas as pd
    import os

    DATA_PATH, ARTIFACT_PATH, _ = get_paths()

    first_sample_token = '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8'  # this is for test

    default_train_file = os.path.join(DATA_PATH, "train.csv")

    df = pd.read_csv(default_train_file)

    for id in df['Id']:

        for p, c in select_annotation_boxes(first_sample_token, level5data):
            print(p)
            print(c)
