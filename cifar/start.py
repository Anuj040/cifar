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
    ["both", "classifier", "pretrain"],
    "Whether to train encoder and classifier in succession or individually.",
)

flags.DEFINE_integer("epochs", 10, "number of training epochs")
flags.DEFINE_integer("train_batch_size", 32, "batchsize for train dataset")
flags.DEFINE_integer("val_batch_size", 32, "batchsize for validation dataset")
flags.DEFINE_float("lr", 0.001, "learning rate for training")


def main(argv):

    model = Cifar()
    model.train(epochs=FLAGS.epochs)


if __name__ == "__main__":
    app.run(main)
