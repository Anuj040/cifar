"""Module implementing custom model.fit for combined model training"""

from textwrap import indent
from typing import Tuple
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow_probability as tfp
from tensorflow.python.ops import state_ops as tf_state_ops


class LossThreshhold(KL.Layer):
    """custom layer for storing moving average of nth percentile of training losses"""

    def __init__(
        self,
        percentile: float = 66.7,
        name: str = "thresh",
        alpha: float = 0.9,
        moving_thresh_initializer: float = 8.0,
        **kwargs
    ):
        """Layer initialization

        Args:
            percentile (float, optional): percentile for thresholding. Defaults to 66.67.
            name (str, optional): name for the tensor. Defaults to "thresh".
            alpha (float, optional): decay value for moving average. Defaults to 0.9.
            moving_thresh_initializer (float, optional): Initial loss threshold. Use experience to tune this.
                                                        Defaults to 8.0
        """
        super().__init__(trainable=False, name=name, **kwargs)
        self.percentile = percentile
        self.moving_thresh_initializer = tf.constant_initializer(
            value=moving_thresh_initializer
        )
        self.alpha = alpha

    def build(self, input_shape):
        """build the layer"""
        shape = (1,)
        self.moving_thresh = self.add_weight(
            shape=shape,
            name="moving_thresh",
            initializer=self.moving_thresh_initializer,
            trainable=False,
        )
        return super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """call method on the layer
        Args:
            inputs (tf.Tensor): sample wise loss values for a given batch

        Returns:
            tf.Tensor (shape = (1,)): loss threshold value for importance sampling
        """
        batch_loss_thresh = tfp.stats.percentile(
            inputs, q=self.percentile, axis=[0], interpolation="linear"
        )
        self.moving_thresh = tf_state_ops.assign(
            self.moving_thresh,
            self.alpha * self.moving_thresh + (1.0 - self.alpha) * batch_loss_thresh,
            # use_locking=self._use_locking,
        )
        return self.moving_thresh

    def get_config(self) -> dict:
        """Setting up the layer config

        Returns:
            dict: config key-value pairs
        """
        base_config = super().get_config()
        config = {
            "alpha": self.alpha,
            "moving_thresh_initializer": self.moving_thresh_initializer,
        }
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """shape of the layer output"""
        return (1,)


class Trainer(KM.Model):  # pylint: disable=too-many-ancestors
    """Custom function for combined training of classifier/autoencoder using model.fit()
        With functionality for selective back propagation for accelerated training.

    Args:
        combined (KM.Model): combined classifier/autoencoder model
    """

    def __init__(self, combined: KM.Model) -> None:
        super().__init__()
        self.combined = combined

        # Extract the classifier model object
        self.classifier_model = KM.Model(
            inputs=self.combined.input,
            outputs=self.combined.get_layer("logits").output,
        )
        self.loss_thresh = LossThreshhold()

    def compile(
        self,
        optimizer: tf.keras.optimizers,
        loss: dict,
        loss_weights: dict,
        metrics: dict,
    ) -> None:
        """compiles the model object with corresponding attributes

        Args:
            optimizer (tf.keras.optimizers): optimizer for model training
            loss (dict): loss definitions for the model outputs
            loss_weights (dict): weights for the loss functions
            metrics (dict): performance metrics for the model outputs
        """
        super().compile()
        self.optimizer = optimizer
        self.loss = loss

        assert len(self.loss) == len(
            loss_weights
        ), "provide weights for all the loss definitions"
        self.loss_weights = loss_weights

        assert len(self.loss) == len(
            metrics
        ), "provide metric functions for all outputs, 'None' wherever not applicable"
        self.loss_metrics = metrics

        # Store the keys for each model output
        ## Make sure that the keys are in the same order as the model outputs
        self.loss_keys = self.loss.keys()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """method to process model call for classification

        Args:
            inputs (tf.Tensor): input tensor

        Returns:
            tf.Tensor: model output
        """

        return self.classifier_model(inputs, training=False)

    def importance_sampler(
        self,
        inputs: tf.Tensor,
        outputs: tf.Tensor,
        losses: tf.Tensor,
        batch_size: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """[summary]

        Args:
            inputs (tf.Tensor): [description]
            outputs (tf.Tensor): [description]
            losses (tf.Tensor): [description]
            batch_size (tf.Tensor): [description]

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: [description]
        """

        # selecting top 1/3rd lossy samples
        # loss_threshold = tfp.stats.percentile(
        #     losses, q=66.67, axis=[0], interpolation="linear"
        # )
        # selecting based on mving average of top 67 percentile loss values
        loss_threshold = self.loss_thresh(losses)

        batch_size = tf.cast(batch_size, tf.int64)[0]

        # get the indices for samples with loss > threshold
        indices = tf.where(losses[:batch_size] >= loss_threshold)

        # batch size for the filtered batch
        batch_size_apparent = tf.shape(indices)[0]
        batch_size_apparent = tf.cast(batch_size_apparent, dtype=tf.float32)

        # indices for the image pairs for contrastive loss
        # TODO: general and special case
        indices = tf.concat(values=[indices, indices + batch_size], axis=0)

        # filtered input-output pairs
        return (
            tf.gather_nd(inputs, indices),
            tuple(tf.gather_nd(output, indices) for output in outputs),
            batch_size_apparent,
        )

    def train_step(self, data: tf.Tensor) -> dict:
        """method to implement a training step on the combined model

        Args:
            data (tf.Tensor): a batch instance consisting of input-output pair

        Returns:
            dict: dict object containing batch performance parameters
        """
        # unpack the data
        images, outputs = data

        # Filter out the most lossy samples
        # get the model outputs
        model_outputs = self.combined(images, training=False)

        losses = 0.0
        # calculate losses
        for i, key in enumerate(self.loss_keys):
            # weighted representative losses for each sample
            losses += (
                self.loss[key](outputs[i], model_outputs[i]) * self.loss_weights[key]
            )

        # get the batch_size
        # for contrastive loss need to pick the image pairs
        # TODO: make it general with contrastive loss as a special case
        batch_size = 0.5 * tf.cast(tf.shape(losses), dtype=tf.float32)

        # if gap between min and max loss large
        # prepare a batch only of high loss samples
        (images, outputs, batch_size_apparent) = tf.cond(
            # gap b/w min & max loss controls the use of selective backprop
            tf.greater(tf.math.reduce_min(losses), 0.20 * tf.math.reduce_max(losses)),
            # true cond.
            lambda: (images, outputs, batch_size),
            # false cond.: pass through importance sampler
            lambda: self.importance_sampler(images, outputs, losses, batch_size),
        )

        # Train the combined model only on the selected samples
        with tf.GradientTape() as tape:

            # get the model outputs
            model_outputs = self.combined(images, training=True)

            losses = []
            losses_grad = []
            # calculate losses
            for i, key in enumerate(self.loss_keys):
                losses.append(self.loss[key](outputs[i], model_outputs[i]))

                # for gradient calculations
                # losses multiplied with corresponding weights
                # tune the learning rate considering the dynamic batch sizing
                losses_grad.append(
                    tf.sqrt(batch_size_apparent / batch_size)
                    * tf.reduce_mean(losses[i])
                    * self.loss_weights[key]
                )

        # calculate and apply gradients
        grads = tape.gradient(
            losses_grad,
            self.combined.trainable_weights,
        )
        self.optimizer.apply_gradients(zip(grads, self.combined.trainable_weights))

        # prepare the logs dictionary
        logs = dict(zip(self.loss_keys, losses))
        logs = {key: tf.reduce_mean(value) for key, value in logs.items()}

        # Add metrics if applicable
        for i, key in enumerate(self.loss_keys):
            metric_func = self.loss_metrics[key]

            # Only evaluate the not-None metrics
            if metric_func is not None:
                metric_func.update_state(outputs[i], model_outputs[i])
                logs[metric_func.name] = metric_func.result()

        # House-keeping
        del tape
        return logs

    @property
    def metrics(self):
        # list `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # Without this property, `reset_states()` will have to called
        # manually.
        return [
            metric_func
            for key, metric_func in self.loss_metrics.items()
            if metric_func is not None
        ]

    def test_step(self, data: tf.Tensor) -> dict:
        """method to implement evaluation step on the classifier model

        Args:
            data (tf.Tensor): a batch instance consisting of input-output pair

        Returns:
            dict: dict object containing batch performance parameters on classification task only
        """
        # unpack the data
        images, outputs = data

        # get the model outputs
        model_outputs = self.classifier_model(images, training=False)

        # Get the metric calculator
        metric_func = self.loss_metrics["logits"]

        # Calculate the performance metrics
        metric_func.update_state(outputs, model_outputs)

        # return the logs dict
        return {metric_func.name: metric_func.result()}


if __name__ == "__main__":
    # trainer = Trainer()
    layer = LossThreshhold()
    for _ in range(10):
        a = tf.random.uniform(shape=(15, 1))
        b = layer(a)
        print(b)
