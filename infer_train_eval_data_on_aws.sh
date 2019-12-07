# Run this script to do test data inference
DATA_ID=15
python3 test_making_inference.py --inference_file /dltraining/test_results/infer_results.pickle \
      --token_file /dltraining/artifacts/lyft_val_token_from_rgb.pickle \
      --pred_file train_val_pred.csv \
      --data_name train \
      --from_rgb_detection