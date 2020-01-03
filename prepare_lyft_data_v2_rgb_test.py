import unittest
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
from prepare_lyft_data_v2 import FrustumGenerator, get_all_boxes_in_single_scene, \
    parse_frustum_point_record, load_train_data
import matplotlib.pyplot as plt
import numpy as np
from test_data_loader import load_test_data
from object_classifier import TLClassifier


class FrustumRGBTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.level5testdata = load_test_data()
        self.object_classifier=TLClassifier()

    def test_one_sample_token(self):
        test_sample_token = self.level5testdata.sample[0]['token']

        print(test_sample_token)

        fg = FrustumGenerator(sample_token=test_sample_token, lyftd=self.level5testdata)

        fp = next(fg.generate_frustums_from_2d(self.object_classifier))

        example = fp.to_train_example()
        example_proto_str = example.SerializeToString()
        example_tensors = parse_frustum_point_record(example_proto_str)

        assert np.allclose(example_tensors['frustum_point_cloud'].numpy(), fp.point_cloud_in_box)

        #assert np.allclose(example_tensors['rot_box_3d'].numpy(), fp._get_rotated_box_3d())  # (8,3))


if __name__ == '__main__':
    unittest.main()
