"""Module implementing custom model.fit for combined model training"""

import tensorflow as tf
import tensorflow.keras.models as KM


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

    def train_step(self, data: tf.Tensor) -> dict:
        """method to implement a training step on the combined model

        Args:
            data (tf.Tensor): a batch instance consisting of input-output pair

        Returns:
            dict: dict object containing batch performance parameters
        """
        images, outputs = data

        # Train the combined model
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
                losses_grad.append(tf.reduce_mean(losses[i]) * self.loss_weights[key])

        # calculate and apply gradients
        grads = tape.gradient(
            losses_grad,
            self.combined.trainable_weights,
        )
        self.optimizer.apply_gradients(zip(grads, self.combined.trainable_weights))

        # prepare the logs dictionary
        logs = dict(zip(self.loss_keys, losses))

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
    trainer = Trainer()
