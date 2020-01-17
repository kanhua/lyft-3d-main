from object_classifier import TLClassifier
from vis_util import draw_bounding_boxes_on_image_array, draw_bounding_box_on_image_array
from skimage.io import imread, imsave
import numpy as np

sample_image_file = "/dltraining/dataset/train_images/host-a004_cam0_1232815253451064006.jpeg"
# sample_image_file = "/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/train_images/host-a004_cam0_1232815252251064006.jpeg"

sample_image = imread(sample_image_file)

tlc = TLClassifier()

nboxes = tlc.detect_multi_object(sample_image, score_threshold=[0.6 for i in range(9)],
                                 rearrange_to_pointnet_convention=False,
                                 target_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9], output_target_class=True)
from model_util import g_type_object_of_interest, map_2d_detector

image_to_be_drawn = np.copy(sample_image)
strings = [[g_type_object_of_interest[map_2d_detector[int(nboxes[i, 5])]]] for i in range(nboxes.shape[0])]
draw_bounding_boxes_on_image_array(image_to_be_drawn, nboxes[:, 0:4], display_str_list_list=strings)

print(nboxes)
imsave("test.png", image_to_be_drawn)
