"""Involution module"""

import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


# pylint: disable=R0913
def involution_model(
    filters: int = 32,
    kernel_size: int = 3,
    strides: int = 1,
    reduction: int = 4,
    group_channels: int = 16,
    name: str = "",
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
    print(out)

    return KM.Model(inputs=input_tensor, outputs=out, name=f"invol_{name}")


def involution(
    filters: int = 32,
    kernel_size: int = 3,
    strides: int = 1,
    reduction: int = 4,
    group_channels: int = 16,
    name: str = "",
):
    """involution module as proposed by Duo Li et al. in
    Involution: Inverting the Inherence of Convolution for Visual Recognition

    Args:
        filters (int): number of filters for the incoming and outgoing tensor. Default 32
        kernel_size (int): kernel size for the involution. Default 3
        strides (int, optional): strides for involution. Defaults to 1.
        reduction (int, optional): channel reduction for dynamic weights calculation. Defaults to 4.
        group_channels (int, optional): number of channels in a group. Defaults to 16.
        name (str, optional): string for layer name. Defaults to ""

    Returns:
        involution function
    """
    # pylint: disable=unnecessary-lambda
    def layer(input_tensor: tf.Tensor) -> tf.Tensor:

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
            strides=[1, strides, strides, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # involution kernels are broadcasted over the "group_channels"
        out = tf.reshape(
            out,
            shape=(
                b_size,
                kernel_size ** 2,
                h_tensor,
                w_tensor,
                groups,
                group_channels,
            ),
        )
        # sum over kernel dimension
        out = tf.reduce_sum(weight * out, axis=1)

        # reshape back to 4D tensor (Batch x Height x width x channels)
        out = tf.reshape(
            out, shape=(b_size, h_tensor, w_tensor, filters), name=f"invol_{name}"
        )
        return out

    return layer


if __name__ == "__main__":
    channels = 128
    a = tf.random.uniform(shape=(2, 16, 16, channels), dtype=tf.float32)

    model = involution(channels, 3, strides=2)
    print(model(a).shape)
