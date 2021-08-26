"""Module implementing custom model.fit for combined model training"""

import tensorflow as tf
import tensorflow.keras.models as KM
import tensorflow_probability as tfp
from cifar.model_utils.custom_layers import NLossThreshhold, BatchMaker

# pylint: disable=too-many-ancestors, too-many-instance-attributes
class Trainer(KM.Model):
    """Custom function for combined training of classifier/autoencoder using model.fit()
        With functionality for selective back propagation for accelerated training.

    Args:
        combined (KM.Model): combined classifier/autoencoder model
    """

    def __init__(self, combined: KM.Model, train_batch_size: int) -> None:
        super().__init__()
        self.combined = combined

        # Extract the classifier model object
        self.classifier_model = KM.Model(
            inputs=self.combined.input,
            outputs=self.combined.get_layer("logits").output,
        )
        self.loss_thresh = NLossThreshhold(max_no_values=10)
        self.batch_maker_layer = BatchMaker(batch_size=train_batch_size)
        self.batch_maker_contrastive_pair = BatchMaker(
            batch_size=train_batch_size, name="select_batch_contrast"
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

    def batch_aggregator(
        self,
        inputs: tf.Tensor,
        gt_outputs: tf.Tensor,
        indices: tf.Tensor,
        batch_size: tf.Tensor,
    ) -> dict:
        """[summary]

        Args:
            inputs (tf.Tensor): [description]
            gt_outputs (tf.Tensor): [description]
            indices (tf.Tensor): indices for the selected samples from a given batch
            batch_size (tf.tensor)

        Returns:
            dict: batch logs
        """

        # filtered input-output pairs
        images, outputs = (
            tf.gather_nd(inputs, indices),
            tuple(tf.gather_nd(output, indices) for output in gt_outputs),
        )
        images_aggregate, outputs_aggregate, train_bool = self.batch_maker_layer(
            [images, outputs]
        )

        # for contrastive loss need to pick the image pairs
        # TODO: general and special case
        # filtered complimentary contrastive (input-output) pairs
        images_contrast, outputs_contrast = (
            tf.gather_nd(inputs, indices + batch_size),
            tuple(tf.gather_nd(output, indices + batch_size) for output in gt_outputs),
        )
        (
            images_contrast_aggregate,
            outputs_contrast_aggregate,
            train_bool,
        ) = self.batch_maker_contrastive_pair([images_contrast, outputs_contrast])

        # Combined batch
        logs = tf.cond(
            train_bool,
            lambda: self.selective_backprop(
                tf.concat([images_aggregate, images_contrast_aggregate], axis=0),
                tuple(
                    tf.concat(
                        [outputs_aggregate[i], outputs_contrast_aggregate[i]], axis=0
                    )
                    for i in range(len(outputs_aggregate))
                ),
                batch_size,
            ),
            lambda: self.skip_train(),
        )

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

    def selective_backprop(
        self, images: tf.Tensor, outputs: tf.Tensor, batch_size: tf.Tensor
    ) -> dict:
        """[summary]

        Args:
            images (tf.Tensor): [description]
            outputs (tf.Tensor): [description]
            batch_size (tf.Tensor): [description]

        Returns:
            dict: [description]
        """
        # Defined batch size
        batch_size = tf.cast(batch_size, tf.float32)

        # Apparent batch_size: slightly different from 'batch size' as selected samples
        # may vary from batch to batch
        # TODO: make it general with contrastive loss as a special case
        batch_size_apparent = 0.5 * tf.cast(tf.shape(images), dtype=tf.float32)

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

        # get the indices for samples with loss > threshold
        indices = tf.where(losses[:batch_size] >= loss_threshold)

        # if gap between min and max loss large
        # prepare a batch only of high loss samples
        logs = tf.cond(
            # gap b/w min & max loss controls the use of selective backprop
            tf.greater(loss_threshold, batch_loss_threshold),
            # true cond.
            lambda: self.skip_train(),
            # false cond.: pass through importance sampler
            lambda: self.batch_aggregator(images, outputs, indices, batch_size),
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
    trainer = Trainer()
