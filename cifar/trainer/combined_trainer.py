"""Module implementing custom model.fit for combined model training"""

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
            moving_thresh_initializer (float): Initial loss threshold. Use experience to tune this.
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
        shape = ()
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
            tf.Tensor (shape = ()): loss threshold value for importance sampling
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
            "percentile": self.percentile,
            "threshhold": self.moving_thresh,
        }
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """shape of the layer output"""
        return ()


class NLossThreshhold(KL.Layer):
    """custom layer for storing history of last "m" batches and returning top
    n-th percentile of the value tensor."""

    def __init__(
        self,
        max_no_values: int,
        name: str = "thresh",
        percentile: float = 66.7,
        **kwargs
    ):
        """Layer initialization

        Args:
            max_no_values (int): Last 'm' number of batches to be stored
            name (str, optional): name for the tensor. Defaults to "thresh".
            percentile (float, optional): percentile for thresholding. Defaults to 66.67.
        """
        super().__init__(trainable=False, name=name, **kwargs)
        self.max_factor = max_no_values
        self.percentile = percentile
        self.moving_thresh_initializer = tf.zeros_initializer()

    def build(self, input_shape):
        """build the layers"""
        self.loss_holder = self.add_weight(
            shape=self.max_factor * input_shape[0],
            name="moving_thresh",
            initializer=self.moving_thresh_initializer,
            trainable=False,
        )
        self.moving_thresh = self.add_weight(
            shape=(),
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
            tf.Tensor (shape = ()): loss threshold value for importance sampling
        """

        temp_loss_holder = tf.concat(values=[self.loss_holder, inputs], axis=0)

        temp_loss_holder = tf.slice(
            temp_loss_holder,
            begin=[tf.shape(inputs)[0]],
            size=[self.max_factor * tf.shape(inputs)[0]],
        )
        self.loss_holder = tf_state_ops.assign(self.loss_holder, temp_loss_holder)

        self.moving_thresh = tf_state_ops.assign(
            self.moving_thresh,
            tfp.stats.percentile(
                self.loss_holder, q=self.percentile, axis=[0], interpolation="linear"
            ),
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
            "max_no_values": self.max_factor,
            "moving_thresh_initializer": self.moving_thresh_initializer,
            "percentile": self.percentile,
            "threshhold": self.moving_thresh,
            "held_values": self.loss_holder,
        }
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """shape of the layer output"""
        return ()


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
        self.loss_thresh = NLossThreshhold(max_no_values=10)

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
        loss_threshold: tf.Tensor,
        batch_size: tf.Tensor,
    ) -> dict:
        """[summary]

        Args:
            inputs (tf.Tensor): [description]
            outputs (tf.Tensor): [description]
            losses (tf.Tensor): [description]
            loss_threshold (tf.Tensor): [description]
            batch_size (tf.tensor)

        Returns:
            dict: batch logs
        """
        # get the indices for samples with loss > threshold
        indices = tf.where(losses[:batch_size] >= loss_threshold)

        # batch size for the filtered batch
        batch_size_apparent = tf.cast(tf.shape(indices)[0], dtype=tf.float32)

        # indices for the image pairs for contrastive loss
        # TODO: general and special case
        indices = tf.concat(values=[indices, indices + batch_size], axis=0)

        # filtered input-output pairs
        images, outputs = (
            tf.gather_nd(inputs, indices),
            tuple(tf.gather_nd(output, indices) for output in outputs),
        )
        batch_size = tf.cast(batch_size, tf.float32)
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

    def skip_train(self) -> dict:
        """method for generating a dummy logs dictionary for skipped batches

        Returns:
            dict:
        """
        # prepare the zero-logs dictionary
        logs = dict(zip(self.loss_keys, [0.0] * len(self.loss_keys)))
        # Add metrics if applicable
        for _, key in enumerate(self.loss_keys):
            metric_func = self.loss_metrics[key]

            # Unpdated metrics value
            if metric_func is not None:
                logs[metric_func.name] = metric_func.result()
        return logs

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
        batch_size = tf.cast(batch_size, tf.int64)[0]

        # Get the batch loss threshold
        batch_loss_threshold = tfp.stats.percentile(
            losses[:batch_size], q=66.67, axis=[0], interpolation="linear"
        )
        # batch_loss_threshold = tf.reduce_max(losses[:batch_size])
        # Get the hitorical threshold
        loss_threshold = self.loss_thresh(losses[:batch_size])

        # if gap between min and max loss large
        # prepare a batch only of high loss samples
        logs = tf.cond(
            # gap b/w min & max loss controls the use of selective backprop
            tf.greater(loss_threshold, batch_loss_threshold),
            # true cond.
            lambda: self.skip_train(),
            # false cond.: pass through importance sampler
            lambda: self.importance_sampler(
                images, outputs, losses, loss_threshold, batch_size
            ),
        )

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
    layer = NLossThreshhold(max_no_values=20, moving_thresh_initializer=7)
    # layer = LossThreshhold()
    for _ in range(100):
        a = 3.0 + 2.0 * tf.random.uniform(shape=(15,))
        b = layer(a)
        print(b)
