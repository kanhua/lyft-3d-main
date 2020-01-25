import unittest

from config_tool import get_paths
import os
import pickle
from vis_util import draw_bounding_boxes_on_image_array, draw_bounding_box_on_image_array
from skimage.io import imread, imsave

from model_util import map_2d_detector, g_type_object_of_interest
import numpy as np

class ImageDetectionTestCase(unittest.TestCase):
    def test_load_pickle_file(self):
        data_path, artifacts_path, _ = get_paths()
        det_path = os.path.join(artifacts_path, "detection")
        file = "host-a101_cam4_1243095627316053006.pickle"
        root, ext = os.path.splitext(file)
        image_file_path = os.path.join(data_path, 'images', root + ".jpeg")
        import pickle
        with open(os.path.join(det_path, file), 'rb') as fp:
            det = pickle.load(fp)

        image_np = imread(image_file_path)

        sel_id = det['scores'] > 0.5
        sel_classes = det['classes'][sel_id]

        strings = [[g_type_object_of_interest[map_2d_detector[int(sel_classes[i])]]] for i in
                   range(sel_classes.shape[0])]

        draw_bounding_boxes_on_image_array(image_np, det['boxes'][sel_id], display_str_list_list=strings)

        imsave("./artifact/test_read_pickle.png", image_np)

        # save_file = os.path.join(det_path, root + ".pickle")


if __name__ == '__main__':
    unittest.main()
