"""module with custom layers"""
from typing import List, Union

import tensorflow as tf
import tensorflow.keras.layers as KL
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
        **kwargs,
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
        **kwargs,
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
        tf_state_ops.assign(self.loss_holder, temp_loss_holder)
        tf_state_ops.assign(
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


class BatchMaker(KL.Layer):
    """custom layer for aggregating samples across batches"""

    def __init__(self, batch_size: int = 32, name: str = "select_batch", **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.batch_size = tf.cast(batch_size, tf.int32)

    def build(self, input_shapes):
        """layer for holding the samples of interest"""

        # Aggregate batch shape
        model_inputs_batch_shape = (self.batch_size,)
        # Looping over all the dimensions
        for i in range(1, len(input_shapes[0])):
            model_inputs_batch_shape = model_inputs_batch_shape + (input_shapes[0][i],)

        self.model_inputs_holder = self.add_weight(
            shape=model_inputs_batch_shape,
            name="model_inputs",
            initializer=tf.zeros_initializer(),
            trainable=False,
        )
        # Outputs list # For mulitple outputs case
        if isinstance(input_shapes[1], (list, tuple)):
            self.model_outputs_holder = []
            for out_num, model_output in enumerate(input_shapes[1]):
                # Aggregate batch shape
                batch_shape = (self.batch_size,)
                # Looping over all the dimensions
                for dim_num in range(1, len(model_output)):
                    batch_shape = batch_shape + (model_output[dim_num],)

                self.model_outputs_holder.append(
                    self.add_weight(
                        shape=batch_shape,
                        name=f"model_outputs_{out_num}",
                        initializer=tf.zeros_initializer(),
                        trainable=False,
                    )
                )
        else:
            # Aggregate batch shape
            batch_shape = (self.batch_size,)
            # Looping over all the dimensions
            for dim_num in range(1, len(input_shapes[1])):
                batch_shape = batch_shape + (input_shapes[1][dim_num],)

            self.model_outputs_holder = self.add_weight(
                shape=batch_shape,
                name="model_outputs",
                initializer=tf.zeros_initializer(),
                trainable=False,
            )

        # Counter for number of samples aggreagated in the current batch being built
        self.fresh_sample_counter = self.add_weight(
            shape=(),
            name="sample_counter",
            initializer=tf.zeros_initializer(),
            trainable=False,
            dtype=self.batch_size.dtype,
        )
        # Boolean for train step execution
        self.run_train_step = self.add_weight(
            shape=(),
            name="train_bool",
            initializer=tf.ones_initializer(),
            trainable=False,
            dtype=tf.bool,
        )
        return super().build(input_shapes)

    def call(
        self, inputs: List[Union[List[tf.Tensor], tf.Tensor]]
    ) -> List[Union[List[tf.Tensor], tf.Tensor]]:
        if not isinstance(inputs, list) or len(inputs) <= 1:
            raise Exception(
                "Batchmaker must be called on a list of tensors "
                "Input 0: Model Inputs, Input 1: Model Outputs. Got: " + str(inputs)
            )
        images = inputs[0]
        model_outputs = inputs[1]
        incoming_batch_size = tf.shape(images)[0]

        start_index = tf.cond(
            self.run_train_step,
            lambda: tf.zeros((), dtype=tf.int32),
            lambda: self.fresh_sample_counter,
        )
        temp_counter = start_index + incoming_batch_size
        end_index = tf.cond(
            tf.greater(temp_counter, self.batch_size),
            lambda: self.batch_size,
            lambda: temp_counter,
        )
        replace_indices = tf.range(start_index, end_index)

        tf.compat.v1.scatter_update(
            self.model_inputs_holder, replace_indices, images[: end_index - start_index]
        )

        # Outputs list # For mulitple outputs case
        if isinstance(model_outputs, (tuple, list)):
            self.model_outputs_holder = [
                tf.compat.v1.scatter_update(
                    self.model_outputs_holder[i],
                    replace_indices,
                    model_outputs[i][: end_index - start_index],
                )
                for i in range(len(model_outputs))
            ]
        else:
            self.model_outputs_holder = tf.compat.v1.scatter_update(
                self.model_outputs_holder,
                replace_indices,
                model_outputs[: end_index - start_index],
            )

        tf_state_ops.assign(self.fresh_sample_counter, temp_counter)
        tf_state_ops.assign(
            self.run_train_step,
            tf.greater_equal(self.fresh_sample_counter, self.batch_size),
        )
        temp_inputs_holder = tf.cond(
            self.run_train_step,
            lambda: tf.concat(
                [
                    self.model_inputs_holder,
                    images[end_index - start_index :],
                ],
                axis=0,
            ),
            lambda: self.model_inputs_holder,
        )
        temp_outputs_holder = tf.cond(
            self.run_train_step,
            lambda: [
                tf.concat(
                    [
                        self.model_outputs_holder[i],
                        model_outputs[i][end_index - start_index :],
                    ],
                    axis=0,
                )
                for i in range(len(model_outputs))
            ]
            if isinstance(model_outputs, (tuple, list))
            else tf.concat(
                [
                    self.model_outputs_holder,
                    model_outputs[end_index - start_index :],
                ],
                axis=0,
            ),
            lambda: self.model_outputs_holder,
        )
        return (
            temp_inputs_holder,
            temp_outputs_holder,
            self.run_train_step,
        )
