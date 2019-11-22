from absl import flags
from absl import app

from config_tool import set_paths

FLAGS = flags.FLAGS
flags.DEFINE_string("model_checkpoint", "", "model checkpoint (ended in model.ckpt)")
flags.DEFINE_string("data_path", "", "data path")
flags.DEFINE_string("artifact_path", "", "artifact path")


def main(argv):
    set_paths(FLAGS.data_path, FLAGS.artifact_path, FLAGS.model_checkpoint)


if __name__ == "__main__":
    app.run(main)
