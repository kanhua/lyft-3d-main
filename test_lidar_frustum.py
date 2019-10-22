import matplotlib.pyplot as plt

from viz_util_for_lyft import draw_lidar_simple
from prepare_lyft_data import get_pc_in_image_fov



def test_draw_lidar_in_fov():
    demo_lidar_sample_data_token = "ec9950f7b5d4ae85ae48d07786e09cebbf4ee771d054353f1e24a95700b4c4af"
    pc_fov,image = get_pc_in_image_fov(demo_lidar_sample_data_token, 'CAM_FRONT')

    draw_lidar_simple(pc_fov)

    input()

    plt.imshow(image)
    plt.show()





test_draw_lidar_in_fov()