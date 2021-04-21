import sys

from cifar.utils.generator import DataGenerator
from cifar.utils.losses import MultiLayerAccuracy

sys.path.append("./")
import tensorflow as tf
import tensorflow.keras.models as KM
from tensorflow.keras.callbacks import Callback


class EvalCallback(Callback):
    """custom callback to evaluate classifier model's performance on validation dataset

    Args:
        model (KM.Model): Model being trained
        val_generator (DataGenerator): validation dataset generator
        layers (int, optional): number of layers to take classification tensor from. Defaults to 1.
    """

    def __init__(self, model: KM.Model, val_generator: DataGenerator, layers: int = 1):
        super(EvalCallback, self).__init__()

        self.val_generator = val_generator
        # Extract model object relevant to evaluate classification performance
        self.evaluator = KM.Model(
            inputs=model.input,
            outputs=model.get_layer("logits").output,
        )
        # Compile the model object with losses and performance metrics
        self.evaluator.compile(loss=None, metrics=MultiLayerAccuracy(layers=layers))

    def on_epoch_end(self, epoch, logs: dict = None):

        metrics = self.evaluator.evaluate(self.val_generator(), verbose=0)
        logs["val_acc"] = metrics[1]
