import unittest

from prepare_lyft_data import extract_single_box, \
    parse_train_csv, level5data, extract_boxed_clouds,get_sample_images
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud


class MyTestCase(unittest.TestCase):

    def test_extract_box(self):
        train_df = parse_train_csv()

        box, sample_data_token = extract_single_box(train_df, 0)

        sample_token = "92bff46db1dbfc9679edc8091770c4256ac3c027e9f0a9c31dfc4fff41f6f677"
        box_from_annotation_token = level5data.get_box(sample_token)

        # self.assertEqual(box,box_from_annotation_token)

    def test_extract_box_clouds(self):
        a, b = extract_boxed_clouds(100)

        print(a.shape)
        print(b.shape)

    def test_get_sample_images(self):

        #train_df = parse_train_csv()

        #sample_token=train_df.iloc[0,0]
        #print(sample_token)
        sample_token = "db8b47bd4ebdf3b3fb21598bb41bd8853d12f8d2ef25ce76edd4af4d04e49341"

        get_sample_images(sample_token)



if __name__ == '__main__':
    unittest.main()
