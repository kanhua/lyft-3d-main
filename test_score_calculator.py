import pandas as pd

from test_making_inference import ScoreCalculator
from config_tool import get_paths
import os
import numpy as np

data_path, _, _ = get_paths()

gt_file = os.path.join(data_path, "train.csv")

pred_file = "train_val_pred.csv"

sc = ScoreCalculator(pred_csv_file=pred_file, gt_csv_file=gt_file)

scores = sc.calculate_single_entry(1)

mean_score = sc.calculate_mean_ious()

#print(scores)
#print(mean_score)

iou_levels=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

for il in iou_levels:
    avg_precision = sc.calculate_average_precision(il)
    print("iou level:{}, avg precision: {}".format(il, np.mean(avg_precision)))
