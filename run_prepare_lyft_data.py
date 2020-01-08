import tensorflow as tf
from prepare_lyft_data_v2 import get_all_boxes_in_single_scene, load_train_data
from test_data_loader import load_test_data

from absl import flags, logging
from absl import app

from config_tool import get_paths
import os
import re

flags.DEFINE_list("scenes", "", "scenes to be processed")
flags.DEFINE_string("data_type", "train", "file type")
flags.DEFINE_boolean("from_rgb", False, "whether from RGB detection")

FLAGS = flags.FLAGS


class SceneProcessor(object):

    def __init__(self, data_type, from_rgb_detection):
        if data_type == "train":
            self.lyftd = load_train_data()
        elif data_type == "test":
            self.lyftd = load_test_data()
        else:
            raise ValueError("invalid data type. Valid dataset names are train or test")
        self.from_rgb_detection = from_rgb_detection
        self.data_type = data_type
        if self.from_rgb_detection:
            from object_classifier import TLClassifier
            self.object_classifier = TLClassifier()
        else:
            self.object_classifier = None

    def process_one_scene(self, scene_num):
        _, artifact_path, _ = get_paths()
        if self.from_rgb_detection:
            file_type = "rgb"
        else:
            file_type = "gt"
        with tf.io.TFRecordWriter(
                os.path.join(artifact_path,
                             "scene_{0}_{1}_{2}.tfrec".format(scene_num, self.data_type, file_type))) as tfrw:
            for fp in get_all_boxes_in_single_scene(scene_num, self.from_rgb_detection,
                                                    self.lyftd, object_classifier=self.object_classifier):
                tfexample = fp.to_train_example()
                tfrw.write(tfexample.SerializeToString())


def list_all_files(data_dir=None, pat="scene_\d+_train.tfrec"):
    if data_dir is None:
        _, artifact_path, _ = get_paths()
        data_dir = artifact_path

    files = []
    for file in os.listdir(data_dir):
        match = re.match(pat, file)
        if match:
            files.append(os.path.join(data_dir, file))

    return files


def main(argv):
    from multiprocessing import Pool
    logging.set_verbosity(logging.INFO)

    sp = SceneProcessor(FLAGS.data_type, from_rgb_detection=FLAGS.from_rgb)

    if "all" in FLAGS.scenes:
        scenes_to_process = range(210)
    else:
        scenes_to_process = map(int, FLAGS.scenes)

    if FLAGS.from_rgb:  # it seems that object detector does not support parallel processes
        for s in scenes_to_process:
            # map(sp.process_one_scene, scenes_to_process)
            sp.process_one_scene(s)
    else:
        logging.info("parallel processing GT data:")
        with Pool(processes=7) as p:
            p.map(sp.process_one_scene, scenes_to_process)


if __name__ == "__main__":
    app.run(main)
