import os

from object_classifier import TLClassifier

from config_tool import get_paths

from skimage.io import imread

import numpy as np

tlc = TLClassifier()

data_path, artifacts_path, _ = get_paths()

image_path = os.path.join(data_path, "images")

from tqdm import tqdm

det_path = os.path.join(artifacts_path, "detection")
if not os.path.exists(det_path):
    os.makedirs(det_path)

for file in tqdm(os.listdir(image_path)):
    if ".jpeg" in file:
        image_np = imread(os.path.join(image_path, file))
        sel_boxes = tlc.detect_multi_object(image_np, output_target_class=True,
                                            rearrange_to_pointnet_convention=True,
                                            score_threshold=[0.4 for i in range(9)],
                                            target_classes=[i for i in range(1, 10, 1)])

        root, ext = os.path.splitext(file)
        np.save(os.path.join(det_path, root + ".npy"), sel_boxes)
