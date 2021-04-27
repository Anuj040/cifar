import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


def channel_attention(features: int, reduction: int = 16, name: str = "") -> KM.Model:
    """channel attention model

    Args:
        features (int): number of features for incoming tensor
        reduction (int, optional): Reduction ratio for the MLP to squeeze information across channels. Defaults to 16.
        name (str, optional): Defaults to "".

    Returns:
        KM.Model: channelwise attention appllier model
    """

    input_tensor = KL.Input(shape=(None, None, features))

    # Average pool over a feature map across channels
    avg = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    # Max pool over a feature map across channels
    max_pool = tf.reduce_max(input_tensor, axis=[1, 2], keepdims=True)

    # Number of features for middle layer of shared MLP
    reduced_features = int(features // reduction)

    dense1 = KL.Dense(reduced_features)
    avg_reduced = dense1(avg)
    max_reduced = dense1(max_pool)

    dense2 = KL.Dense(features)
    avg_attention = dense2(KL.Activation("relu")(avg_reduced))
    max_attention = dense2(KL.Activation("relu")(max_reduced))

    # Channel-wise attention
    overall_attention = KL.Activation("sigmoid")(avg_attention + max_attention)

    return KM.Model(
        inputs=input_tensor, outputs=input_tensor * overall_attention, name=name
    )


def spatial_attention(
    features: int, kernel: int = 7, bias: bool = False, name: str = ""
) -> KM.Model:
    """spatial attention model

    Args:
        features (int): number of features for incoming tensor
        kernel (int): convolutional kernel size
        bias (bool, optional): whether to use bias in convolutional layer
        name (str, optional): Defaults to "".

    Returns:
        KM.Model: spatial attention appllier model
    """

    input_tensor = KL.Input(shape=(None, None, features))
    # Average pool across channels for a given spatial location
    avg = tf.reduce_mean(input_tensor, axis=[-1], keepdims=True)

    # Max pool across channels for a given spatial location
    max_pool = tf.reduce_max(input_tensor, axis=[-1], keepdims=True)

    concat_pool = tf.concat([avg, max_pool], axis=-1)

    # Attention for spatial locations
    conv = KL.Conv2D(
        1, (kernel, kernel), strides=(1, 1), padding="same", use_bias=bias
    )(concat_pool)
    attention = KL.Activation("sigmoid")(KL.BatchNormalization()(conv))

    return KM.Model(inputs=input_tensor, outputs=input_tensor * attention, name=name)


def cbam_block(
    input_tensor: tf.Tensor,
    features: int,
    kernel: int = 7,
    spatial: bool = False,
    name: str = "",
) -> tf.Tensor:
    """Convolutional Block Attention Module as proposed by Woo et. al in
    CBAM: Convolutional Block Attention Module

    Args:
        input_tensor (tf.Tensor):
        features (int): number of features for incoming layer
        kernel (int): kernel size for spatial attention module
        spatial (bool, optional): whether to apply spatial attention. Defaults to False.
                                False: equivalent to squueze and excitation block with both max and avg. pool.
        name (str, optional): Defaults to "".

    Returns:
        tf.Tensor: [description]
    """
    out_tensor = channel_attention(features, name=name + "chn")(input_tensor)
    if spatial:
        out_tensor = spatial_attention(features, kernel=kernel, name=name + "spt")(
            out_tensor
        )
    return out_tensor


if __name__ == "__main__":
    import numpy as np

    a = np.random.uniform(size=(10, 8, 8, 4)).astype(np.float32)
    a_out = cbam_block(a, 4, 3, True, name="block1")
