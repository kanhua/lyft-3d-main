# 3D Object detection with Lyft pointnet data

This project adapted the code of [frustum-pointnet](http://stanford.edu/~rqi/frustum-pointnets/) to solve this past Kaggle competition [Lyft 3D Object Detection for Autonomous Vehicles](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles)


## Main workflow of training and testing

### Setting the path configuration

```bash
python setup_path.py --model_checkpoint /path/to/model/checkpt --data_path /path/to/data --artifact_path /path/to/artifact --object_detection_model_path /object/detection/model/path
```
artifact path are the paths that store user-generated data.


### Training the point clouds by frustum pointnetV1

- Prepare the frustum pointnet data of selected scenes in Lyft train data:
```bash
python run_prepare_lyft_data.py --scenes scene_1,scene_2,... --data_type {train, test} [--from_rgb]
```

    - ```--data_type```: sets the type of the data. 
    - If ```--from_rgb``` is set, the frustum will be generated by a 2D object detector. If ```--from_rgb``` was not set, the frustum is selected by ground truths.

To preprocess all scenes:
```bash
python run_prepare_lyft_data.py --scenes all
```
Or use scripts ```prepare_data_from_rgb.sh``` or ```prepare_data_from_gt.sh```.


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

- Run ```infer_train_eval_data.sh```
- Run ```test_score_calculator.py```

To generate the file for Kaggle submission, run ```merge_prediction.py```

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


## Visualization

Bird view frustums: [Visualize frustum pipelines.ipynb](Visualize frustum pipelines.ipynb)

Visualize predicted results in 3D: [pred_viewer_test.py](pred_viewer_test.py)

Plot frustum in 3D: [plot_v2_data.py](plot_v2_data.py)

Test plot frustums and rotated frustums point cloud points: ```prepare_lyft_data_v2_test.test_plot_one_frustum```

