from absl import flags
from absl import app

from config_tool import set_paths

FLAGS = flags.FLAGS
flags.DEFINE_string("model_checkpoint", "", "model checkpoint (ended in model.ckpt)")
flags.DEFINE_string("data_path", "", "data path")
flags.DEFINE_string("test_data_path", "", "data path")
flags.DEFINE_string("artifact_path", "", "artifact path")
flags.DEFINE_string("object_detection_model_path", "", "object detection model path")


def main(argv):
    set_paths(FLAGS.data_path, FLAGS.test_data_path, FLAGS.artifact_path, FLAGS.model_checkpoint,
              FLAGS.object_detection_model_path)


if __name__ == "__main__":
    app.run(main)
