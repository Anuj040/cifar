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


if __name__ == "__main__":
    categorical_focal_loss()
