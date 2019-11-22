DATA_DIR="/dltraining/artifacts/lyft_val_4.pickle"
MODEL_LOG_DIR="/dltraining/log_v1"
python train.py --gpu 0 --model frustum_pointnets_v1 --log_dir $MODEL_LOG_DIR --num_point 1024 --max_epoch 5 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --data_dir $DATA_DIR
