DATA_FILE="/dltraining/artifacts/scene_0_test.tfrec"
MODEL_LOG_DIR="/dltraining/log_v1_test/"
RESTORE_MODEL_PATH="/dltraining/log_v1_test//model.ckpt"

python train_v2.py --gpu 0 --model frustum_pointnets_v1 --log_dir $MODEL_LOG_DIR \
                  --max_epoch 5 --batch_size 32 --decay_step 800000 \
                  --decay_rate 0.5 --data_dir $DATA_FILE

#--restore_model_path $RESTORE_MODEL_PATH \
