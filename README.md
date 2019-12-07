## Workflow of training the model on AWS

1. set up link of the dataset, using ```link_file_on_aws.sh```
2. Run ```python prepare_lyft_data.py```
2. run ```bash lyft_train_v1_on_aws.sh```. Note that this has to be bash because of for-loop is used in this script.
3. Download data from the AWS instance, using ```aws-scripts/download_model.sh```
4. Unzip and extract the model file to this folder as ```./log_v1```
5. Run inference using ```sh infer_train_eval_data.sh```
6. See visualized anaysis in  ```Check train-val data detection.ipynb``` (This file is not finished yet.)


## Workflow of running inference on AWS

1. Run ```test_prepare_data_from_rgb_detection.py``` to run RGB detection and generate pickle data
2. Run frusutm-point net inference: ```run_pointnet_inference_on_aws.sh```
3. Generate pred csv data: ```infer_train_eval_data_on_aws.sh``` 