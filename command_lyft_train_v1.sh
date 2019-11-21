DATA_DIR="/Users/kanhua/Dropbox/Programming/lyft-3d-main/artifact/lyft_val_3_0.pickle"

python train.py --gpu 0 --model frustum_pointnets_v1 --log_dir log_v1 --num_point 1024 --max_epoch 5 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --data_dir $DATA_DIR --no_intensity
