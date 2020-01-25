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
from test_data_loader import load_test_data

lyftd = load_test_data()

import pickle


def load_image_paths_in_scene(scene_num):
    with open(os.path.join(det_path, "test_scene_{}_images.pickle".format(scene_num)), 'rb') as fp:
        all_images = pickle.load(fp)
        return all_images


def detect_image_in_scene(scene_num):
    print("processing :{}".format(scene_num))
    pre_processed_scene_image_file = os.path.join(det_path, "test_scene_{}_images.pickle".format(scene_num))
    if os.path.exists(pre_processed_scene_image_file):
        print("use proprocessed paths")
        all_images = load_image_paths_in_scene(scene_num)
    else:
        all_images = [ip for ip in get_all_image_paths_in_single_scene(scene_number=scene_num, ldf=lyftd)]

    for file in tqdm(all_images):
        head, tail = os.path.split(file)
        root, ext = os.path.splitext(tail)
        save_file = os.path.join(det_path, root + ".pickle")

        tlc.detect_and_save(image_path=os.path.join(image_dir, file), save_file=save_file)


scene_num = range(28, 40, 1)

for s in scene_num:
    detect_image_in_scene(s)
