DATA_FILE="/dltraining/artifacts/"
TRAINED_MODEL_PATH="/dltraining/log_v1_test/model.ckpt"

python test_v2.py --gpu 0 --model frustum_pointnets_v1 \
                  --batch_size 32 --model_path $TRAINED_MODEL_PATH\
                  --data_dir $DATA_FILE \
