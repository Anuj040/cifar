import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

LARGE_NUM = 1e9


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

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

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


def multi_layer_focal(
    weight: float = 1.0, alpha: float = 0.25, gamma: float = 2.0, layers: int = 1
):
    """custom layer to calculate focal loss on logits taken from multiple latent vectors

    Args:
        weight (float, optional): weighing factor. Defaults to 1.0.
        alpha (float, optional): [description]. Defaults to 0.25.
        gamma (float, optional): focusing parameter for modulating factor (1-p) for focal loss. Defaults to 2.0.
        layers (int, optional): number of layers to take classification tensor from. Defaults to 1.

    Returns:
        function: loss calculating function
    """
    # focal loss calculating function
    focal = categorical_focal_loss(alpha=alpha, gamma=gamma)

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

        y_true = tf.split(y_true, num_or_size_splits=layers, axis=-1)
        # Split the logits from different levels
        y_pred = tf.split(y_pred, num_or_size_splits=layers, axis=-1)
        loss = 0.0

        for i in range(len(y_true)):
            loss += focal(y_true[i], y_pred[i])

        return weight * loss

    _loss.__name__ = "mlti"

    return _loss


class MultiLayerAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name: str = "acc", layers: int = 1, **kwargs):
        """custom keras metric layer to calculate accuracy with predicted labels from multiple latent vectors

        Args:
            name (str, optional): Layer name. Defaults to "acc".
            layers (int, optional): number of layers to take classification tensor from. Defaults to 1.
        """
        super(MultiLayerAccuracy, self).__init__(name=name, **kwargs)
        self.layers = layers

        # Stores the summation of metric value over the whole dataset
        self.metric = self.add_weight(name="true_count", initializer="zeros")

        # Samples count
        self.metric_count = self.add_weight(name="Count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.split(y_true, num_or_size_splits=self.layers, axis=-1)
        # Split the logits from different levels
        y_pred = tf.split(
            tf.expand_dims(y_pred, axis=-1), num_or_size_splits=self.layers, axis=1
        )
        true_logits = y_true[0]
        # Predicted label by taking an elementwise maximum across all layers
        pred_logits = tf.reduce_max(tf.concat(y_pred, axis=2), axis=2)

        # Get true and pred labels
        true_labels = tf.argmax(true_logits, axis=-1)
        pred_labels = tf.argmax(pred_logits, axis=-1)

        correct_pred = true_labels == pred_labels
        correct_pred = tf.cast(correct_pred, tf.float32)

        # Number of samples in a given batch
        count = tf.cast(K.shape(correct_pred)[0], self.dtype)
        # Sum of metric value for the processed samples
        self.metric.assign_add(K.sum(correct_pred))
        # Total number of samples processed
        self.metric_count.assign_add(count)

    def result(self) -> tf.Tensor:
        # Average metric value
        return self.metric / self.metric_count

    def reset_states(self):
        # metric state reset at the start of each epoch.
        self.metric.assign(0.0)
        self.metric_count.assign(0)


def contrastive_loss(
    hidden_norm: bool = True, temperature: float = 1.0, weights: float = 1.0
):
    """method for implementing contrastive loss

    Args:
        hidden_norm (bool, optional): whether or not to use normalization on the hidden vector. Defaults to True.
        temperature (float, optional): temperature scaling. Defaults to 1.0.
        weights (float, optional): weighing factor. Defaults to 1.0.

    Returns:
        loss function.
    """

    def _loss(y_true: tf.Tensor, hidden: tf.Tensor) -> tf.Tensor:
        """
        Args:
        hidden (tf.Tensor): hidden vector of shape (2*bsz, dim).
        """
        # Get (normalized) hidden1 and hidden2.
        if hidden_norm:
            hidden = tf.math.l2_normalize(hidden, -1)
        # Split the feature vectors between Augmentation1 and Augmentation2 of the same image
        hidden1, hidden2 = tf.split(hidden, 2, 0)
        batch_size = tf.shape(hidden1)[0]

        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        # Cosine distance between the feature vectors from Augmentation1
        logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
        # Remove the contribution from the cosine distance of the vector with self
        logits_aa = logits_aa - masks * LARGE_NUM

        # Cosine distance between the feature vectors from Augmentation2
        logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
        # Remove the contribution from the cosine distance of the vector with self
        logits_bb = logits_bb - masks * LARGE_NUM

        # Cosine distance between the feature vectors from Augmentation1 and Augmentation2
        logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
        # Cosine distance between the feature vectors from Augmentation2 and Augmentation1
        logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature

        loss_a = tf.losses.CategoricalCrossentropy(from_logits=True)(
            labels, tf.concat([logits_ab, logits_aa], 1)
        )
        loss_b = tf.losses.CategoricalCrossentropy(from_logits=True)(
            labels, tf.concat([logits_ba, logits_bb], 1)
        )
        loss = tf.reduce_mean(loss_a + loss_b)
        loss = weights * loss

        return loss

    _loss.__name__ = "con"

    return _loss


if __name__ == "__main__":
    categorical_focal_loss()
