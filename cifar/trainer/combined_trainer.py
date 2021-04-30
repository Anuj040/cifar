"""Module implementing custom model.fit for combined model training"""

import tensorflow as tf
import tensorflow.keras.models as KM


class Trainer(KM.Model):  # pylint: disable=too-many-ancestors
    """Custom function for combined training of classifier/autoencoder using model.fit()
        With functionality for selective back propagation for accelerated training.

    Args:
        combined (KM.Model): combined classifier/autoencoder model
    """

    def __init__(self, combined: KM.Model):
        super().__init__()
        self.combined = combined

    def compile(
        self,
        optimizer: tf.keras.optimizers,
        loss: dict,
        loss_weights: dict,
        metrics: dict,
    ):
        super().compile()

    def call(self):
        ...

    def train_step(self, inputs):
        ...


if __name__ == "__main__":
    trainer = Trainer()
