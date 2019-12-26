## Main workflow of training and testing

### Training the point clouds by frustum pointnetV1

- Prepare the frustum pointnet data of selected scenes in Lyft train data:
```bash
python run_prepare_lyft_data.py --scenes scene_1,scene_2,...
```
To preprocess all scenes:
```bash
python run_prepare_lyft_data.py --scenes all
```

- Train the model
Make model setting in ```lyft_train_v2_on_local.sh``` or ```lyft_train_v2_on_aws.sh```. Then execute the scripts.
```bash
sh lyft_train_v2_on_local.sh
```

- Run inference with the trained model
```bash
sh lyft_test_v2_on_local.sh
```
So far, the program searches the pattern ```scene_\d+_train.tfrec``` in the designated directories assigned 
in ```lyft_train_v2_on_local.sh``` and ```lyft_test_v2_on_local.sh```.

#### Not completed yet:
Use the function ```parse_pointnet_output.get_box_from_inference()``` to transform the inferred results back to world coordinates. 
This needs two steps:
1. Correct the frustum angle:
- The predicted heading angle
- The predicted center

2. Transform the predicted corners from camera coordinates to world coordinates.

### Training 2D object detector

#### Install Tensorflow object detection API
TODO: installation instruction.


#### Prepare the training data
This 2D object detector uses Tensorflow object detection API. It was tested on tensorflow 1.14, 
and does not support Tensorflow 2.0 yet.
- Prepare the object detection data to match the format for Tensorflow object detection API.
In the last line of ```prepare_object_detection_data.py```, change ```write_data_to_files(param)```.
```param``` is the number of sample tokens in ```train.csv``` to be run through.
Set ```param=None``` to go through all sample tokens.
```bash
python prepare_object_detection_data.py.
```

#### Train the model

The model settings are in ```./object_detection_models/models```. 
Configure the model in ````XXXXXX.config````, ```run_model.sh```, and then run ```run_model.sh```.
The category file is ```object_detecton_models/models/lyft_object_map.pbtxt```.

### Export the model
Run ```export_model.sh```.
TODO: the path setting somehow does not allow the model to be used on local machine, 
this may due to that the path setting in the configuration files are absolute rather than relative.  

### Run inference

TODO: Full scripts to detect all data not completed yet. 
- Set the object detection model path in ```user_config.txt```. See ```default_config.txt``` for examples.
- See ```parse_pointnet_output.py``` for an example of running object detection


## Workflow of training the model on AWS

1. set up link of the dataset, using ```link_file_on_aws.sh```
2. Run ```python prepare_lyft_data.py```
2. run ```bash lyft_train_v1_on_aws.sh```. Note that this has to be bash because of for-loop is used in this script.
3. Download data from the AWS instance, using ```aws-scripts/download_model.sh```
4. Unzip and extract the model file to this folder as ```./log_v1```
5. Run inference using ```sh infer_train_eval_data.sh```
6. See visualized analysis in  ```Check train-val data detection.ipynb``` (This file is not finished yet.)


## Workflow of running inference on AWS

1. Run ```test_prepare_data_from_rgb_detection.py``` to run RGB detection and generate pickle data
2. Run frusutm-point net inference: ```run_pointnet_inference_on_aws.sh```
3. Generate pred csv data: ```infer_train_eval_data_on_aws.sh``` 