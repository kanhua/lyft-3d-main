PROGRAM_PATH=/dltraining/models/research
PIPELINE_CONFIG_PATH=/home/ec2-user/lyft-3d-main/object_detecton_models/models/faster_rcnn_resnet101_kitti/faster_rcnn_resnet101_lyft.config
MODEL_DIR=/dltraining/lyft_object_detection_models/models/faster_rcnn_resnet101/model_log
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python $PROGRAM_PATH/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostder