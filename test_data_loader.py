"""
Prepare test data

"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle
from typing import Tuple, List
from PIL import Image

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer

import matplotlib.pyplot as plt


def load_test_data():
    DATA_PATH = '/Users/kanhua/Downloads/3d-object-detection-test'
    level5data_snapshot_file = "level5testdata.pickle"

    if os.path.exists(os.path.join(DATA_PATH, level5data_snapshot_file)):
        with open(os.path.join(DATA_PATH, level5data_snapshot_file), 'rb') as fp:
            level5testdata = pickle.load(fp)
    else:

        level5testdata = LyftDataset(data_path='/Users/kanhua/Downloads/3d-object-detection-test',
                                     json_path='/Users/kanhua/Downloads/3d-object-detection-test/data/',
                                     verbose=True)
        with open(os.path.join(DATA_PATH, level5data_snapshot_file), 'wb') as fp:
            pickle.dump(level5testdata, fp)

    print("number of scenes:", len(level5testdata.scene))

    return level5testdata
