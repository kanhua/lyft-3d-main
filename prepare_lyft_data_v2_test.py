from prepare_lyft_data_v2 import FrustumGenerator, get_all_boxes_in_single_scene
from prepare_lyft_data import level5data
import matplotlib.pyplot as plt
import numpy as np


def test_one_sample_token():
    test_sample_token = level5data.sample[0]['token']

    print(test_sample_token)

    fg = FrustumGenerator(sample_token=test_sample_token, lyftd=level5data)

    fp = next(fg.generate_frustums())


def test_plot_one_frustum():
    test_sample_token = level5data.sample[0]['token']

    print(test_sample_token)

    fg = FrustumGenerator(sample_token=test_sample_token, lyftd=level5data)

    fig, ax = plt.subplots(1, 2)
    prev_cam_token=None
    for fp in fg.generate_frustums():
        if prev_cam_token==None:
            prev_cam_token=fp.camera_token
            fp.render_image(ax[0])
        elif prev_cam_token!=fp.camera_token:
            break

        fp.render_point_cloud_on_image(ax[0])

        fp.render_point_cloud_top_view(ax[1])

        prev_cam_token=fp.camera_token

    plt.show()


def test_one_scene():
    get_all_boxes_in_single_scene(0, False, level5data)


test_one_sample_token()
# test_one_scene()
test_plot_one_frustum()
