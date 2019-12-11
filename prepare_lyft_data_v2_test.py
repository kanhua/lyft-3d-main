from prepare_lyft_data_v2 import FrustumGenerator,get_all_boxes_in_single_scene
from prepare_lyft_data import level5data


def test_one_sample_token():

    test_sample_token=level5data.sample[0]['token']

    print(test_sample_token)

    fg=FrustumGenerator(sample_token=test_sample_token,lyftd=level5data)

    fg.generate_frustums()

def test_one_scene():

    get_all_boxes_in_single_scene(0,False,level5data)



test_one_sample_token()
test_one_scene()