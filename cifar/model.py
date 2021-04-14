import sys

sys.path.append("./")
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from absl import flags

from cifar.utils.generator import DataGenerator
from cifar.utils.losses import categorical_focal_loss

FLAGS = flags.FLAGS


class Cifar:
    def __init__(self) -> None:
        self.model = self.build()
        self.classifier = self.build_classify(num_classes=10)

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
        self.encoder_model = self.encoder(features=encoder_features, name="encoder")

        # build the decoder model
        decoder = self.decoder(features=decoder_features, name="decoder")

        input_tensor = KL.Input(
            shape=(32, 32, 3)
        )  # shape of images for cifar10 dataset

        # Encode the images
        encoded = self.encoder_model(input_tensor)
        # Decode the image
        decoded = decoder(encoded)

        return KM.Model(inputs=input_tensor, outputs=decoded, name="AutoEncoder")

    def build_classify(self, num_classes: int = 10) -> KM.Model:
        """Method to build classifier model

        Returns:
            KM.Model: Classifier model
        """
        input_tensor = KL.Input(shape=(32, 32, 3))
        encoded_features = self.encoder_model(input_tensor)

        encoded_flat = KL.Flatten()(encoded_features)
        probs = KL.Dense(num_classes, activation="sigmoid", name="logits")(encoded_flat)

        return KM.Model(inputs=input_tensor, outputs=probs, name="classifier")

    def compile(self, classifier_loss: str = "focal"):
        """method to compile the model object with optimizer, loss definitions and metrics

        Args:
            classifier_loss (str, optional): loss function to use for classifier training. Defaults to "focal".

        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics="mae"
        )
        cce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        )
        focal = categorical_focal_loss()
        self.classifier.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=focal if classifier_loss == "focal" else cce,
            metrics="accuracy",
        )

    def train(self, epochs: int = 10, classifier_loss: str = "focal"):
        """method to initiate model training

        Args:
            epochs (int, optional): total number of training epochs. Defaults to 10.
            classifier_loss(str, optional): loss function for classifier training. Defaults to "focal
        """

        # compile the model object
        self.compile(classifier_loss=classifier_loss)
        if FLAGS.train_mode in ["both", "pretrain"]:
            # prepare the generator
            train_generator = DataGenerator(shuffle=True, train_mode="pretrain")
            # number of trainig steps per epoch
            train_steps = len(train_generator)
            self.model.fit(
                train_generator(),
                initial_epoch=0,
                epochs=epochs,
                workers=8,
                verbose=2,
                steps_per_epoch=train_steps,
            )

        if FLAGS.train_mode in ["both", "classifier"]:

            # prepare the generators for classifier training
            train_generator_classifier = DataGenerator(
                split="train", shuffle=True, train_mode="classifier"
            )
            val_generator_classifier = DataGenerator(
                split="val", train_mode="classifier"
            )

            # number of trainig steps per epoch
            train_steps = len(train_generator_classifier)

            if FLAGS.train_mode == "both":
                # Use feature representations learnt from AutoEncoder training
                self.encoder_model.trainable = False

            self.classifier.fit(
                train_generator_classifier(),
                initial_epoch=0,
                epochs=epochs,
                workers=8,
                verbose=2,
                steps_per_epoch=train_steps,
                validation_data=val_generator_classifier(),
            )


if __name__ == "__main__":
    model = Cifar()
    model.train()
