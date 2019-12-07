# From tensorflow/models/research/
PROGRAM_PATH=/dltraining/models/research
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=faster_rcnn_resnet101_lyft.config
TRAINED_CKPT_PREFIX=/dltraining/lyft_object_detection_models/models/faster_rcnn_resnet101/model_log/model.ckpt-9377
EXPORT_DIR=/dltraining/lyft_object_detection_models/models/faster_rcnn_resnet101/model_export/
python $PROGRAM_PATH/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}