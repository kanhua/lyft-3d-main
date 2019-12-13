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

    ax_dict = {}
    for fp in fg.generate_frustums():
        if fp.camera_token not in ax_dict.keys():
            fig, ax = plt.subplots(1, 3)
            ax_dict[fp.camera_token] = (fig, ax)
            fp.render_image(ax[0])
        else:
            ax = ax_dict[fp.camera_token][1]

        fp.render_point_cloud_on_image(ax[0])

        fp.render_point_cloud_top_view(ax[1])

        fp.render_rotated_point_cloud_top_view(ax[2])

    for key in ax_dict.keys():
        fig, ax = ax_dict[key]
        channel = level5data.get("sample_data", key)['channel']
        fig.savefig("./artifact/{}.png".format(channel))


def test_one_scene():
    get_all_boxes_in_single_scene(0, False, level5data)


test_one_sample_token()
# test_one_scene()
test_plot_one_frustum()
