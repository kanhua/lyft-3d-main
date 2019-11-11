import pandas as pd

sample_csv="/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/sample_submission.csv"

pred_csv="/Users/kanhua/Dropbox/Programming/lyft-3d-main/test_pred.csv"

sample_df=pd.read_csv(sample_csv,delimiter=',')

pred_df=pd.read_csv(pred_csv,delimiter=",")



