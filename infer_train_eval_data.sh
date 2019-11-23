# Run this script to do test data inference
DATA_ID=17
python3 /Users/kanhua/Downloads/frustum-pointnets/train/test.py \
        --data_path /Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/artifacts/lyft_frustum_$DATA_ID.pickle \
        --model_path /Users/kanhua/Downloads/frustum-pointnets/train/log_v1/model.ckpt \
        --batch_size 32 \
        --dump_result \
        --output /Users/kanhua/Downloads/lyft-3d-results
python3 test_making_inference.py --inference_file /Users/kanhua/Downloads/lyft-3d-results/infer_results.pickle \
      --token_file /Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/artifacts/lyft_frustum_token_$DATA_ID.pickle \
      --pred_file train_val_pred.csv \
      --data_name train