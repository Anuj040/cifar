import os
import sys

sys.path.append("./")
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from absl import flags
from keras_drop_block import DropBlock2D

from cifar.utils.generator import DataGenerator
from cifar.utils.losses import multi_layer_accuracy, multi_layer_focal

FLAGS = flags.FLAGS


def deconv_block(input_tensor: tf.Tensor, features: int, name: str) -> tf.Tensor:
    """Bottleneck style deconvolutional block. Applies convolution to decrease to fewer features.
    upscales in lower feature space. Convols to specified feature numbers

    Args:
        input_tensor (tf.Tensor): tensor to apply deconvolution to
        features (int): number of features
        name (str): name for the tensors

    Returns:
        tf.Tensor: output tensor
    """
    out = input_tensor

    out = KL.Conv2D(
        int(features // 2),
        1,
        strides=(1, 1),
        name=name + f"_c{1}",
    )(input_tensor)
    out = KL.Activation("relu")(KL.BatchNormalization()(out))

    out = KL.Conv2DTranspose(
        int(features // 2),
        (4, 4),
        strides=(2, 2),
        padding="same",
        name=name + f"_d",
    )(out)
    out = KL.Activation("relu")(KL.BatchNormalization()(out))

    out = KL.Conv2D(
        features,
        1,
        strides=(1, 1),
        name=name + f"_c{2}",
    )(out)
    out = KL.Activation("relu")(KL.BatchNormalization()(out))

    return out


class Cifar:
    def __init__(self) -> None:
        if FLAGS.train_mode in ["both", "pretrain"]:
            self.model = self.build()
        if FLAGS.train_mode in ["both", "classifier"]:
            self.classifier = self.build_classify(num_classes=10)
        if FLAGS.train_mode == "combined":
            self.combined = self.build_combined(num_classes=10)

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
            name=name + f"_conv_{1}",
        )(input_tensor)
        encoded = KL.Activation("relu")(KL.BatchNormalization()(encoded))
        encoded_list = [encoded]
        for i, feature_num in enumerate(features[1:], start=2):
            encoded = KL.Conv2D(
                feature_num,
                3,
                strides=(2, 2),
                name=name + f"_conv_{i}",
            )(encoded)
            encoded = KL.Activation("relu")(KL.BatchNormalization()(encoded))
            encoded_list.append(encoded)
        return KM.Model(inputs=input_tensor, outputs=encoded_list, name=name)

    def decoder(self, features=[8], name="decoder") -> KM.Model:
        """Creates a decoder model object

        Args:
            features (list, optional): list of features in successive hidden layers. Defaults to [8].
            name (str, optional): name for the model object. Defaults to "decoder".

        Returns:
            KM.Model: Decoder model
        """
        input_tensor = KL.Input(shape=(1, 1, features[0]))

        # mismatch between input and output image shape
        padding = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
        padded = tf.pad(
            input_tensor,
            padding,
            "REFLECT" if len(features) < 4 else "SYMMETRIC",
        )

        decoded = padded

        for i, feature_num in enumerate(features[1:], start=1):
            decoded = deconv_block(
                decoded, feature_num, name + f"_deconv_{len(features)-i}"
            )

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
        decoded = DropBlock2D(block_size=5, keep_prob=0.8)(decoded)
        return KM.Model(inputs=input_tensor, outputs=decoded, name=name)

    def build(self) -> KM.Model:
        """Method to build Autoencoder model

        Returns:
            KM.Model: Autoencoder model
        """
        # Number of features in successive hidden layers of encoder
        encoder_features = [64, 128, 256, 512]

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
        decoded = decoder(encoded[-1])

        return KM.Model(inputs=input_tensor, outputs=decoded, name="AutoEncoder")

    def build_classify(self, num_classes: int = 10) -> KM.Model:
        """Method to build classifier model

        Returns:
            KM.Model: Classifier model
        """
        input_tensor = KL.Input(shape=(32, 32, 3))
        # if FLAGS.train_mode == "classifier":
        #     # Use the pretrained encoder for classifier only training
        #     self.encoder_model = KM.load_model("ae_model/ae_model.h5").get_layer(
        #         "encoder"
        #     )

        # Number of features in successive hidden layers of encoder
        encoder_features = [64, 128, 256, 512]

        # build the encoder model
        self.encoder_model = self.encoder(features=encoder_features, name="encoder")
        encoded_features = self.encoder_model(input_tensor)

        # logits from the final layer of features of auto-encoder
        encoded_flat = KL.Flatten()(encoded_features[-1])
        encoded_flat = KL.Dropout(rate=0.2)(encoded_flat)
        probs = KL.Dense(num_classes, activation="sigmoid", name="logits_n")(
            encoded_flat
        )

        # logits from the all but last layer of features of auto-encoder
        pooled_probs = [
            KL.Dense(num_classes, activation="sigmoid", name=f"logits_{i}")(
                KL.Dropout(rate=0.2)(
                    KL.Flatten()(KL.GlobalAveragePooling2D()(features))
                )
            )
            for i, features in enumerate(encoded_features[:-1], start=1)
        ]

        # concatenated logits from all the layers
        probs = KL.Lambda(lambda x: tf.concat(x, axis=-1), name="logits")(
            [probs] + pooled_probs
        )
        return KM.Model(inputs=input_tensor, outputs=probs, name="classifier")

    def build_combined(self, num_classes: int = 10) -> KM.Model:
        """Method to build combined AE-classifier model

        Returns:
            KM.Model: AE-Classifier model
        """
        # Number of features in successive hidden layers of encoder
        encoder_features = [64, 128, 256, 512]

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
        encoded_features = self.encoder_model(input_tensor)
        # Decode the image from the final layer features of Auto-encoder
        decoded = decoder(encoded_features[-1])

        # logits from the final layer of features of auto-encoder
        encoded_flat = KL.Flatten()(encoded_features[-1])
        encoded_flat = KL.Dropout(rate=0.2)(encoded_flat)
        probs = KL.Dense(num_classes, activation="sigmoid", name="logits_n")(
            encoded_flat
        )

        # logits from the all but last layer of features of auto-encoder
        pooled_probs = [
            KL.Dense(num_classes, activation="sigmoid", name=f"logits_{i}")(
                KL.Dropout(rate=0.2)(
                    KL.Flatten()(KL.GlobalAveragePooling2D()(features))
                )
            )
            for i, features in enumerate(encoded_features[:-1], start=1)
        ]

        # concatenated logits from all the layers
        probs = KL.Lambda(lambda x: tf.concat(x, axis=-1), name="logits")(
            [probs] + pooled_probs
        )

        return KM.Model(inputs=input_tensor, outputs=[decoded, probs], name="combined")

    def compile(self, classifier_loss: str = "focal"):
        """method to compile the model object with optimizer, loss definitions and metrics

        Args:
            classifier_loss (str, optional): loss function to use for classifier training. Defaults to "focal".

        """
        if FLAGS.train_mode in ["both", "pretrain"]:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    # Linear lr scaling, with batch size (default = 32)
                    learning_rate=FLAGS.lr
                    * (FLAGS.train_batch_size / 32)
                ),
                loss="mse",
                metrics="mae",
            )
        cce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        )
        focal = multi_layer_focal()
        accuracy = multi_layer_accuracy()
        if FLAGS.train_mode in ["both", "classifier"]:
            self.classifier.compile(
                optimizer=tf.keras.optimizers.Adam(
                    # Linear lr scaling, with batch size (default = 32)
                    learning_rate=FLAGS.lr
                    * (FLAGS.train_batch_size / 32)
                ),
                loss=focal if classifier_loss == "focal" else cce,
                metrics=accuracy,
            )
        if FLAGS.train_mode == "combined":
            self.combined.compile(
                optimizer=tf.keras.optimizers.Adam(
                    # Linear lr scaling, with batch size (default = 32)
                    learning_rate=FLAGS.lr
                    * (FLAGS.train_batch_size / 32)
                ),
                loss={
                    "decoder": "mse",
                    "logits": focal if classifier_loss == "focal" else cce,
                },
                metrics={"decoder": None, "logits": accuracy},
                loss_weights={"decoder": 1.0, "logits": 20.0},
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
            train_generator = DataGenerator(
                batch_size=FLAGS.train_batch_size,
                augment=True,
                shuffle=True,
                cache=FLAGS.cache,
                train_mode="pretrain",
            )
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

            # Save the trained autoencoder model
            os.makedirs("./ae_model", exist_ok=True)
            self.model.save("ae_model/ae_model.h5")

        if FLAGS.train_mode in ["both", "classifier"]:

            # prepare the generators for classifier training
            train_generator_classifier = DataGenerator(
                batch_size=FLAGS.train_batch_size,
                split="train",
                augment=True,
                cache=FLAGS.cache,
                shuffle=True,
                train_mode="classifier",
            )
            val_generator_classifier = DataGenerator(
                batch_size=FLAGS.val_batch_size,
                split="val",
                cache=FLAGS.cache,
                train_mode="classifier",
            )

            # number of trainig steps per epoch
            train_steps = len(train_generator_classifier)

            # if FLAGS.train_mode == "both":
            #     # Use feature representations learnt from AutoEncoder training
            #     self.encoder_model.trainable = True

            self.classifier.fit(
                train_generator_classifier(),
                initial_epoch=0,
                epochs=epochs,
                workers=8,
                verbose=2,
                steps_per_epoch=train_steps,
                validation_data=val_generator_classifier(),
            )

        if FLAGS.train_mode == "combined":

            # prepare the generators for classifier training
            train_generator = DataGenerator(
                batch_size=FLAGS.train_batch_size,
                split="train",
                augment=True,
                shuffle=True,
                cache=FLAGS.cache,
                train_mode="combined",
            )
            val_generator = DataGenerator(
                batch_size=FLAGS.val_batch_size,
                split="val",
                cache=FLAGS.cache,
                train_mode="combined",
            )

            # number of trainig steps per epoch
            train_steps = len(train_generator)

            self.combined.fit(
                train_generator(),
                initial_epoch=0,
                epochs=epochs,
                workers=8,
                verbose=2,
                steps_per_epoch=train_steps,
                validation_data=val_generator(),
            )


if __name__ == "__main__":
    model = Cifar()
    model.train()
