"""
This file tests how the results are put together to be Box3D and the run evaluations


"""

import numpy as np
from parse_pointnet_output import read_frustum_pointnet_output
from prepare_lyft_data import level5data
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D, get_ious
from lyft_dataset_sdk.utils.data_classes import Box


def box_to_box3D(box: Box, sample_token: str):
    box3d = Box3D(sample_token=sample_token, translation=box.center,
                  size=box.wlh, rotation=box.orientation.q, name=box.name, score=box.score)

    return box3d


inference_pickle_file = "/Users/kanhua/Downloads/frustum-pointnets/train/test_results.pickle"
token_pickle_file = "/Users/kanhua/Dropbox/Programming/lyft-3d-main/artifact/lyft_val_token.pickle"

pred_boxes, gt_boxes, sample_token_list = read_frustum_pointnet_output(level5data,
                                                                       inference_pickle_file=inference_pickle_file,
                                                                       token_pickle_file=token_pickle_file)
pred_boxes_3d = []
gt_boxes_3d = []

for i in range(len(pred_boxes)):
    pred_boxes_3d.append(box_to_box3D(pred_boxes[i], sample_token_list[i]))
    gt_boxes_3d.append(box_to_box3D(gt_boxes[i], sample_token_list[i]))

for pbox in pred_boxes_3d:
    ious = get_ious(gt_boxes_3d, pbox)
    print(np.array(ious).max())

