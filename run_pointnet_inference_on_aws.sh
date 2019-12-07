python test.py --data_path /dltraining/artifacts/lyft_val_from_rgb.pickle \
               --model_path /dltraining/log_v1/model.ckpt \
               --batch_size 32 --dump_result --from_rgb_detection --no_intensity \
               --output /dltraining/test_results/