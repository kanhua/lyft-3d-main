import pandas as pd

from test_making_inference import ScoreCalculator

gt_file = "/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/train.csv"

pred_file = "/Users/kanhua/Dropbox/Programming/lyft-3d-main/train_val_pred.csv"

sc = ScoreCalculator(pred_csv_file=pred_file, gt_csv_file=gt_file)


scores=sc.calculate_single_entry(1)

print(scores)
