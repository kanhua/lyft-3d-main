import os
from prepare_lyft_data import prepare_frustum_data_from_scenes, level5data

output_file = os.path.join("./artifact/lyft_val_from_rgb.pickle")
token_file = os.path.join("./artifact/lyft_val_token_from_rgb.pickle")
# prepare_frustum_data_from_traincsv(64, output_file)
prepare_frustum_data_from_scenes(32, output_file, token_filename=token_file, scenes=range(2),
                                 from_rgb_detection=True, lyftdf=level5data)
