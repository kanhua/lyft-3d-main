import os

from object_classifier import FastClassifer

from config_tool import get_paths

from skimage.io import imread

import numpy as np

from tqdm import tqdm

tlc = FastClassifer()

data_path, artifacts_path, _ = get_paths()

image_dir = os.path.join(data_path, "images")

det_path = os.path.join(artifacts_path, "detection")
if not os.path.exists(det_path):
    os.makedirs(det_path)

from prepare_lyft_data_v2 import load_train_data, get_all_image_paths_in_single_scene

train_lyft = load_train_data()
all_images = [ip for ip in get_all_image_paths_in_single_scene(scene_number=170, ldf=train_lyft)]

for file in tqdm(all_images):
    if ".jpeg" in file:
        root, ext = os.path.splitext(file)
        save_file = os.path.join(det_path, root + ".pickle")

        tlc.detect_and_save(image_path=os.path.join(image_dir, file), save_file=save_file)
