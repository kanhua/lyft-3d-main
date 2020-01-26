import os
import pickle
from config_tool import get_paths
from tqdm import tqdm
from absl import app, flags,logging
from prepare_lyft_data_v2 import load_train_data, get_all_image_paths_in_single_scene
from test_data_loader import load_test_data
from object_classifier import FastClassifer

flags.DEFINE_list("scenes", "", "scenes to be processed")
flags.DEFINE_string("data_type", "train", "file type")

FLAGS = flags.FLAGS


def load_image_paths_in_scene(scene_num, det_path, file_pat):
    with open(os.path.join(det_path, file_pat.format(scene_num)), 'rb') as fp:
        all_images = pickle.load(fp)
        return all_images


def detect_image_in_scene(scene_num, lyftd, tlc, file_pat):
    data_path, artifacts_path, _ = get_paths()

    image_dir = os.path.join(data_path, "images")

    det_path = os.path.join(artifacts_path, "detection")
    if not os.path.exists(det_path):
        os.makedirs(det_path)

    logging.info("Run 2D detector on scene number :{}".format(scene_num))

    pre_processed_scene_image_file = os.path.join(det_path, file_pat.format(scene_num))
    if os.path.exists(pre_processed_scene_image_file):
        logging.info("use proprocessed paths")
        all_images = load_image_paths_in_scene(scene_num, det_path, file_pat)
    else:
        all_images = [ip for ip in get_all_image_paths_in_single_scene(scene_number=scene_num, ldf=lyftd)]

    for file in tqdm(all_images):
        head, tail = os.path.split(file)
        root, ext = os.path.splitext(tail)
        save_file = os.path.join(det_path, root + ".pickle")

        tlc.detect_and_save(image_path=os.path.join(image_dir, file), save_file=save_file)


def main(argv):
    logging.set_verbosity(logging.INFO)

    tlc = FastClassifer()

    if FLAGS.data_type == "test":
        lyftd = load_test_data()
        file_pat = "test_scene_{}_images.pickle"
    elif FLAGS.data_type == "train":
        lyftd = load_train_data()
        file_pat = "train_scene_{}_images.pickle"
    else:
        raise ValueError("data_type should be either test or train")

    scene_num = map(int, FLAGS.scenes)

    for s in scene_num:
        detect_image_in_scene(s, lyftd, tlc, file_pat)


if __name__ == "__main__":
    app.run(main)
