import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


class Cifar:
    def __init__(self) -> None:
        self.model = self.build()

    def encoder(self, features=[8], name="encoder") -> KM.Model:
        """Creates an encoder model object

        Args:
            features (list, optional): list of features in successive hidden layers. Defaults to [8].
            name (str, optional): name for the model object. Defaults to "encoder".

        Returns:
            KM.Model: Encoder model
        """
        input_tensor = KL.Input(
            shape=(32, 32, 3)
        )  # shape of images for cifar10 dataset
        encoded = KL.Conv2D(
            features[0],
            3,
            strides=(2, 2),
            use_bias=True,
            activation="relu",
            name=name + f"_conv_{0+1}",
        )(input_tensor)
        for i, feature_num in enumerate(features[1:]):
            encoded = KL.Conv2D(
                feature_num,
                3,
                strides=(2, 2),
                use_bias=True,
                activation="relu",
                name=name + f"_conv_{i+2}",
            )(encoded)
        return KM.Model(inputs=input_tensor, outputs=encoded, name=name)

    def decoder(self, features=[8], name="decoder") -> KM.Model:
        """Creates a decoder model object

        Args:
            features (list, optional): list of features in successive hidden layers. Defaults to [8].
            name (str, optional): name for the model object. Defaults to "decoder".

        Returns:
            KM.Model: Decoder model
        """
        input_tensor = KL.Input(shape=(None, None, features[0]))

        # mismatch between input and output image shape
        padding = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
        padded = tf.pad(
            input_tensor,
            padding,
            "REFLECT",
        )
        decoded = KL.Conv2D(
            features[0],
            3,
            strides=1,
            padding="same",
            use_bias=True,
            activation="relu",
            name=name + f"_deconv_{len(features)}",
        )(padded)
        for i, feature_num in enumerate(features[1:]):
            decoded = KL.Conv2DTranspose(
                feature_num,
                (4, 4),
                strides=(2, 2),
                padding="same",
                use_bias=True,
                activation="relu",
                name=name + f"_deconv_{len(features)-i -1}",
            )(decoded)

        # Final reconstruction back to the original image size
        decoded = KL.Conv2DTranspose(
            3,
            (4, 4),
            strides=(2, 2),
            padding="same",
            kernel_initializer="he_normal",
            use_bias=True,
            activation="tanh",
            name=name + f"_out",
        )(decoded)
        return KM.Model(inputs=input_tensor, outputs=decoded, name=name)

    def build(self) -> KM.Model:
        """Method to build Autoencoder model

        Returns:
            KM.Model: Autoencoder model
        """
        # Number of features in successive hidden layers of encoder
        encoder_features = [4, 8]

        # For decoder number of features in opposite order of encoder
        decoder_features = encoder_features.copy()
        decoder_features.reverse()

        # build the encoder model
        encoder = self.encoder(features=encoder_features, name="encoder")

        # build the decoder model
        decoder = self.decoder(features=decoder_features, name="decoder")

        input_tensor = KL.Input(
            shape=(32, 32, 3)
        )  # shape of images for cifar10 dataset

        # Encode the images
        encoded = encoder(input_tensor)
        # Decode the image
        decoded = decoder(encoded)

        return KM.Model(inputs=input_tensor, outputs=decoded, name="AutoEncoder")


if __name__ == "__main__":
    model = Cifar()
