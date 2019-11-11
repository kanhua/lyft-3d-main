"""
This file tests how the results are put together to be Box3D and the run evaluations


"""

import numpy as np
from parse_pointnet_output import read_frustum_pointnet_output
#from prepare_lyft_data import level5data
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D, get_ious
from lyft_dataset_sdk.utils.data_classes import Box
from typing import List
from test_data_loader import level5testdata


def box_to_box3D(box: Box, sample_token: str):
    box3d = Box3D(sample_token=sample_token, translation=box.center,
                  size=box.wlh, rotation=box.orientation.q, name=box.name, score=box.score)

    return box3d


def write_output_csv(pred_boxes: List[Box], sample_token_list, output_csv_file: str):
    with open(output_csv_file, 'w') as fp:
        fp.write("Id,PredictionString\n")
        prev_sample_token = None
        for idx, sample_token in enumerate(sample_token_list):
            # x, y, z, w, l, h, yaw, c
            score = pred_boxes[idx].score
            x, y, z = pred_boxes[idx].center
            w, l, h = pred_boxes[idx].wlh
            yaw = pred_boxes[idx].orientation.angle * np.sign(pred_boxes[idx].orientation.axis[2])
            c = pred_boxes[idx].name

            pred_string = "{0} {1} {2} {3} {4} {5} {6} {7} {8} ".format(score, x, y, z, w, l, h, yaw, c)

            if sample_token != prev_sample_token:
                if prev_sample_token is not None:
                    fp.write("\n")
                fp.write(sample_token)
                fp.write(",")
            fp.write(pred_string)

            prev_sample_token = sample_token


inference_pickle_file = "/Users/kanhua/Downloads/frustum-pointnets/train/test_results.pickle"
token_pickle_file = "/Users/kanhua/Dropbox/Programming/lyft-3d-main/artifact/lyft_val_token_from_rgb.pickle"
pred_csv_file = "./test_pred.csv"
FROM_RGB_DETECTION = True

pred_boxes, gt_boxes, sample_token_list = read_frustum_pointnet_output(level5testdata,
                                                                       inference_pickle_file=inference_pickle_file,
                                                                       token_pickle_file=token_pickle_file,
                                                                       from_rgb_detection=FROM_RGB_DETECTION)
write_output_csv(pred_boxes, sample_token_list, pred_csv_file)

if not FROM_RGB_DETECTION:
    pred_boxes_3d = []
    gt_boxes_3d = []

    for i in range(len(pred_boxes)):
        pred_boxes_3d.append(box_to_box3D(pred_boxes[i], sample_token_list[i]))
        gt_boxes_3d.append(box_to_box3D(gt_boxes[i], sample_token_list[i]))

    for pbox in pred_boxes_3d:
        ious = get_ious(gt_boxes_3d, pbox)
        print(np.array(ious).max())
