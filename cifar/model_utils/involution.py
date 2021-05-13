"""Involution module"""

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


def involution(
    filters: int,
    kernel_size: int,
    strides: int = 1,
    reduction: int = 4,
    group_channels: int = 16,
) -> KM.Model:
    """involution module as proposed by Duo Li et al. in
    Involution: Inverting the Inherence of Convolution for Visual Recognition

    Args:
        filters (int): [description]
        kernel_size (int): [description]
        strides (int, optional): [description]. Defaults to 1.
        reduction (int, optional): [description]. Defaults to 4.
        group_channels (int, optional): [description]. Defaults to 16.

    Returns:
        KM.Model: [description]
    """
    # pylint: disable=unnecessary-lambda
    input_tensor = KL.Input(shape=(None, None, filters))

    # number of groups
    groups = filters // group_channels

    if strides > 1:
        avgpool = KL.AveragePooling2D(pool_size=(strides, strides))
    else:
        avgpool = KL.Lambda(lambda x: tf.identity(x))

    # weights for involution kernel
    weight = KL.Conv2D(filters // reduction, kernel_size=1, use_bias=False)(
        avgpool(input_tensor)
    )
    weight = KL.Conv2D(
        kernel_size ** 2 * groups,
        kernel_size=1,
        strides=1,
    )(KL.Activation("relu")(KL.BatchNormalization()(weight)))

    # Batch x Height x width x channels
    b_size, h_tensor, w_tensor, _ = tf.shape(weight)
    weight = tf.reshape(
        weight, shape=(b_size, kernel_size ** 2, h_tensor, w_tensor, groups, 1)
    )

    # tensor patches for each involution kernel
    out = tf.image.extract_patches(
        input_tensor,
        sizes=[1, kernel_size, kernel_size, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="SAME",
    )

    # involution kernels are broadcasted over the "group_channels"
    out = tf.reshape(
        out,
        shape=(b_size, kernel_size ** 2, h_tensor, w_tensor, groups, group_channels),
    )
    # sum over kernel dimension
    out = tf.reduce_sum(weight * out, axis=1)

    # reshape back to 4D tensor (Batch x Height x width x channels)
    out = tf.reshape(out, shape=(b_size, h_tensor, w_tensor, filters))

    return KM.Model(inputs=input_tensor, outputs=out)


if __name__ == "__main__":
    a = tf.random.uniform(shape=(2, 224, 224, 32), dtype=tf.float32)

    model = involution(32, 7)
    model.summary()
    print(model(a))
