import numpy as np
from prepare_lyft_data import level5data, parse_train_csv, \
    get_train_data_sample_token_and_box, transform_box_from_world_to_sensor_coordinates

train_df = parse_train_csv()

train_idx = 0

sample_token, train_sample_box = get_train_data_sample_token_and_box(train_idx, train_df)

first_train_sample = level5data.get('sample', sample_token)

sample_data_token = first_train_sample['data']['LIDAR_TOP']

box_in_velo_coordinate = transform_box_from_world_to_sensor_coordinates(train_sample_box, sample_data_token, )

print(box_in_velo_coordinate.center)

print(box_in_velo_coordinate.wlh)