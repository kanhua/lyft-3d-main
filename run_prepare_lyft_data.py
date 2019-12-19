import tensorflow as tf
from prepare_lyft_data_v2 import get_all_boxes_in_single_scene
from prepare_lyft_data import level5data

from absl import flags,logging
from absl import app

flags.DEFINE_list("scenes", "", "scenes to be processed")

FLAGS = flags.FLAGS
from config_tool import get_paths
import os


def process_one_scene(scene_num):
    print("writing one scene:")
    _, artifact_path, _ = get_paths()
    with tf.io.TFRecordWriter(os.path.join("scene_{}_test.tfrec".format(scene_num))) as tfrw:
        for fp in get_all_boxes_in_single_scene(scene_num, False, level5data):
            tfexample = fp.to_train_example()
            tfrw.write(tfexample.SerializeToString())


def main(argv):
    from multiprocessing import Pool

    logging.set_verbosity(logging.INFO)
    scenes_to_process = map(int, FLAGS.scenes)

    with Pool(processes=1) as p:
        p.map(process_one_scene, scenes_to_process)


if __name__ == "__main__":
    app.run(main)
