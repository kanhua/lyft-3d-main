# Run this script to do test data inference
python3 test_prepare_data_from_rgb_detection.py
python3 /Users/kanhua/Downloads/frustum-pointnets/train/test.py --data_path /Users/kanhua/Dropbox/Programming/lyft-3d-main/artifact/lyft_val_from_rgb.pickle --model_path /Users/kanhua/Downloads/frustum-pointnets/train/log_v1/model.ckpt --batch_size 32 --dump_result --from_rgb_detection
python3 test_making_inference.py