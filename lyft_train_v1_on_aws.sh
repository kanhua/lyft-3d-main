DATA_DIR="/dltraining/artifacts/lyft_frustum_0.pickle"
MODEL_LOG_DIR="/dltraining/log_v1"
RESTORE_MODEL_PATH="/dltraining/log_v1/model.ckpt"
python train.py --gpu 0 --model frustum_pointnets_v1 --log_dir $MODEL_LOG_DIR --num_point 1024 \
--max_epoch 5 --batch_size 32 --decay_step 800000 \
--decay_rate 0.5 --data_dir $DATA_DIR --restore_model_path $RESTORE_MODEL_PATH