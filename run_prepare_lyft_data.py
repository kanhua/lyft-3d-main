import tensorflow as tf
from prepare_lyft_data_v2 import get_all_boxes_in_single_scene, load_train_data

from absl import flags, logging
from absl import app

from config_tool import get_paths
import os

flags.DEFINE_list("scenes", "", "scenes to be processed")

FLAGS = flags.FLAGS

level5data = load_train_data()


def process_one_scene(scene_num):
    print("writing one scene:")
    _, artifact_path, _ = get_paths()
    with tf.io.TFRecordWriter(os.path.join(artifact_path, "scene_{}_train.tfrec".format(scene_num))) as tfrw:
        for fp in get_all_boxes_in_single_scene(scene_num, False, level5data):
            tfexample = fp.to_train_example()
            tfrw.write(tfexample.SerializeToString())


def list_all_files(data_dir=None):
    if data_dir is None:
        _, artifact_path, _ = get_paths()
        data_dir = artifact_path

    pat = "scene_\d+_train.tfrec"
    import re
    files = []
    for file in os.listdir(data_dir):
        match = re.match(pat, file)
        if match:
            files.append(os.path.join(data_dir, file))

    return files


def main(argv):
    from multiprocessing import Pool
    logging.set_verbosity(logging.INFO)
    if "all" in FLAGS.scenes:
        scenes_to_process = range(210)
    else:
        scenes_to_process = map(int, FLAGS.scenes)

    with Pool(processes=7) as p:
        p.map(process_one_scene, scenes_to_process)


if __name__ == "__main__":
    app.run(main)
