import os
from prepare_lyft_data import prepare_frustum_data_from_scenes, level5data
from test_data_loader import level5testdata
output_file = os.path.join("./artifact/lyft_val_from_rgb.pickle")
token_file = os.path.join("./artifact/lyft_val_token_from_rgb.pickle")
# prepare_frustum_data_from_traincsv(64, output_file)
prepare_frustum_data_from_scenes(112992, output_file, token_filename=token_file, scenes=range(218),
                                 from_rgb_detection=True, lyftdf=level5testdata)
