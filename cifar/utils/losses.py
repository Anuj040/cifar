import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def categorical_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """Softmax version of focal loss.

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Args:
        alpha (float, optional): [description]. Defaults to 0.25.
        gamma (float, optional): focusing parameter for modulating factor (1-p). Defaults to 2.0.

    Returns:
        function: loss calculating function
    """

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor):

        """focal loss calculating function

        Args:
            y_true (tf.Tensor): A tensor of the same shape as `y_pred`, holding ground truth values
            y_pred (tf.Tensor): predicted logits

        Returns:
            [type]: [description]
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate Focal Loss
        loss = -alpha * y_true * K.pow(1 - y_pred, gamma) * K.log(y_pred) - alpha * (
            1 - y_true
        ) * K.pow(y_pred, gamma) * K.log(1 - y_pred)

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    _loss.__name__ = "focal"

    return _loss


def multi_layer_focal(weight: float = 1.0):
    """custom layer to calculate focal loss on logits taken from multiple latent vectors

    Args:
        weight (float, optional): weighing factor. Defaults to 1.0.

    Returns:
        function: loss calculating function
    """
    # focal loss calculating function
    focal = categorical_focal_loss()

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

        y_true = tf.split(y_true, num_or_size_splits=4, axis=-1)
        # Split the logits from different levels
        y_pred = tf.split(y_pred, num_or_size_splits=4, axis=-1)
        loss = 0.0

        for i in range(len(y_true)):
            loss += focal(y_true[i], y_pred[i])

        return weight * loss

    _loss.__name__ = "mlti"

    return _loss


def multi_layer_accuracy():
    """custom layer to calculate accuracy with predicted labels from multiple latent vectors

    Returns:
        function: metric calculating function
    """
    # Metrics function
    m = tf.keras.metrics.CategoricalAccuracy()

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

        y_true = tf.split(y_true, num_or_size_splits=4, axis=-1)
        # Split the logits from different levels
        y_pred = tf.split(tf.expand_dims(y_pred, axis=-1), num_or_size_splits=4, axis=1)
        true_label = y_true[0]
        # Predicted label by taking an elementwise maximum across all layers
        pred_label = tf.reduce_max(tf.concat(y_pred, axis=2), axis=2)

        m.update_state(true_label, pred_label)

        return m.result()

    _loss.__name__ = "acc"

    return _loss


if __name__ == "__main__":
    categorical_focal_loss()
