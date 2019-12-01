from prepare_lyft_data import get_paths
import pandas as pd
import os

from prepare_object_detection_data import select_annotation_boxes,level5data

def test_run_through_sample_token():


    DATA_PATH, ARTIFACT_PATH, _ = get_paths()

    first_sample_token = '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8'  # this is for test

    default_train_file = os.path.join(DATA_PATH, "train.csv")

    df = pd.read_csv(default_train_file)

    for id in df['Id']:

        for p, c in select_annotation_boxes(first_sample_token, level5data):
            print(p)
            print(c)