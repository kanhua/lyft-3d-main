import tensorflow as tf

tf.compat.v1.enable_eager_execution()
from prepare_lyft_data_v2 import FrustumGenerator, get_all_boxes_in_single_scene, parse_frustum_point_record
from prepare_lyft_data import level5data
import matplotlib.pyplot as plt
import numpy as np

from viz_util_for_lyft import draw_lidar_simple,draw_gt_boxes3d


def test_plot_one_frustum():
    test_sample_token = level5data.sample[0]['token']

    print(test_sample_token)

    fg = FrustumGenerator(sample_token=test_sample_token, lyftd=level5data)

    ax_dict = {}
    fp = next(fg.generate_frustums())

    pc = fp.point_cloud_in_box

    fig=draw_lidar_simple(pc,color=fp.seg_label.astype(np.int)*255)
    draw_gt_boxes3d([fp.box_3d_pts],fig)


if __name__ == "__main__":
    test_plot_one_frustum()
    input()
