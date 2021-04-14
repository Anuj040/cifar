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


def main(argv):

    model = Cifar()
    model.train()


if __name__ == "__main__":
    app.run(main)
