import pandas as pd

from test_making_inference import ScoreCalculator
from config_tool import get_paths
import os

data_path, _, _ = get_paths()

gt_file = os.path.join(data_path,"train.csv")

pred_file = "train_val_pred.csv"

sc = ScoreCalculator(pred_csv_file=pred_file, gt_csv_file=gt_file)

scores = sc.calculate_single_entry(1)

mean_score = sc.calculate_mean_ious()

print(scores)

print(mean_score)
