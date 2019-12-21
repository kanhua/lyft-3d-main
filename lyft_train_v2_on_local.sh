DATA_FILE="/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/artifacts"
MODEL_LOG_DIR="./log_v1_test/"
RESTORE_MODEL_PATH="./log_v1_test/model.ckpt"

python train_v2.py --gpu 0 --model frustum_pointnets_v1 --log_dir $MODEL_LOG_DIR \
                  --max_epoch 200 --batch_size 32 --decay_step 800000 \
                  --decay_rate 0.5 --data_dir $DATA_FILE \
                  --restore_model_path $RESTORE_MODEL_PATH


#--restore_model_path $RESTORE_MODEL_PATH \
