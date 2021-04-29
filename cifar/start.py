import os

# os.environ["TFDS_DATA_DIR"] = "/workspace/tensorflow_datasets"
import sys

sys.path.append("./")
from absl import app, flags

from cifar.model import Cifar

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "train_mode",
    "both",
    ["both", "classifier", "pretrain", "combined"],
    "Whether to train encoder and classifier in succession or individually.",
)
flags.DEFINE_enum(
    "mode", "train", ["train", "eval", "infer"], "Run training, evaluation or inference"
)
flags.DEFINE_integer("epochs", 10, "number of training epochs")
flags.DEFINE_integer("train_batch_size", 32, "batchsize for train dataset")
flags.DEFINE_integer("val_batch_size", 32, "batchsize for validation dataset")
flags.DEFINE_float("gamma", 2.0, "focussing parameter for focal loss")
flags.DEFINE_float("lr", 0.001, "learning rate for training")
flags.DEFINE_bool("cache", False, "whether to store the train/val data in cache")
flags.DEFINE_string("model_path", None, "Model file to resume training from")
flags.DEFINE_string("img_path", None, "Image file to be inferred")


def main(argv):

    model = Cifar(FLAGS.model_path, train_mode=FLAGS.train_mode)
    if FLAGS.mode == "eval":
        model.eval(val_batch_size=FLAGS.val_batch_size)
    elif FLAGS.mode == "infer":
        assert FLAGS.img_path is not None, "\nProvide path to image\n"
        model.infer(FLAGS.img_path)
    else:
        model.custom_train(
            epochs=FLAGS.epochs,
            train_batch_size=FLAGS.train_batch_size,
            val_batch_size=FLAGS.val_batch_size,
            lr=FLAGS.lr,
            gamma=FLAGS.gamma,
            cache=FLAGS.cache,
        )


if __name__ == "__main__":
    app.run(main)
