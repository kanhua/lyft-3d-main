DATA_DIR="/dltraining/artifacts/lyft_frustum_0.pickle"
MODEL_LOG_DIR="/dltraining/log_v1"
RESTORE_MODEL_PATH="/dltraining/log_v1/model.ckpt"
for VALUE in {0..15} # leave three sets of data for evaluation
do
  data_file="/dltraining/artifacts/lyft_frustum_"$VALUE".pickle"
  echo $data_file
  python train.py --gpu 1 --model frustum_pointnets_v1 --log_dir $MODEL_LOG_DIR --num_point 1024 \
                  --max_epoch 5 --batch_size 32 --decay_step 800000 \
                  --decay_rate 0.5 --data_dir $data_file --restore_model_path $RESTORE_MODEL_PATH
done
zip -r /dltraining/log_v1_temp.zip /dltraining/log_v1/

