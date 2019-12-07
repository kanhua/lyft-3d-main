from object_classifier import TLClassifier
from vis_util import draw_bounding_boxes_on_image_array,draw_bounding_box_on_image_array
from skimage.io import imread,imsave
import numpy as np


sample_image_file="/dltraining/dataset/train_images/host-a004_cam0_1232815253451064006.jpeg"


sample_image=imread(sample_image_file)

tlc=TLClassifier()

nboxes=tlc.detect_multi_object(sample_image,score_threshold=[0.6,0.8,0.8],rearrange_to_pointnet_convention=False)

image_to_be_drawn=np.copy(sample_image)
draw_bounding_boxes_on_image_array(image_to_be_drawn,nboxes[:,0:4])

imsave("test.png",image_to_be_drawn)
