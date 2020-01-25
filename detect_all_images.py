import os

from object_classifier import FastClassifer

from config_tool import get_paths

from skimage.io import imread

import numpy as np

from tqdm import tqdm

tlc = FastClassifer()

data_path, artifacts_path, _ = get_paths()

image_path = os.path.join(data_path, "images")

det_path = os.path.join(artifacts_path, "detection")
if not os.path.exists(det_path):
    os.makedirs(det_path)

for file in tqdm(os.listdir(image_path)):
    if ".jpeg" in file:
        root, ext = os.path.splitext(file)
        save_file = os.path.join(det_path, root + ".pickle")

        tlc.detect_and_save(image_path=os.path.join(image_path, file), save_file=save_file)
