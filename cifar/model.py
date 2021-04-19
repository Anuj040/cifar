import os
import re
import sys
from typing import List, Tuple

from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint

sys.path.append("./")
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from absl import flags
from keras_drop_block import DropBlock2D

from cifar.utils.callbacks import EvalCallback
from cifar.utils.generator import DataGenerator
from cifar.utils.losses import MultiLayerAccuracy, contrastive_loss, multi_layer_focal

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


def conv_block(
    input_tensor: tf.Tensor, skip_tensors: List[tf.Tensor], features: int, name: str
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    out = KL.Conv2D(
        features,
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name=name + f"_c{1}",
    )(input_tensor)
    out = KL.BatchNormalization()(out)

    skip_tensors_next = []
    for i, feature_tensor in enumerate(skip_tensors, start=1):
        feature_tensor = KL.AveragePooling2D(pool_size=(2, 2), strides=2)(
            feature_tensor
        )
        skip_tensors_next.append(feature_tensor)
    skip_connection = tf.concat(skip_tensors_next, axis=-1)

    out = KL.Activation("relu", name=name + "_relu")(out + skip_connection)
    skip_tensors_next.append(out)
    return out, skip_tensors_next


class Cifar:
    def __init__(self, model_path: str) -> None:
        """

        Args:
            model_path (str): path to the saved model to resume training from
        """
        self.model_path = model_path

        # Number of features in successive hidden layers of encoder
        self.encoder_features = [128, 256, 512, 1024]

        if FLAGS.train_mode in ["both", "pretrain"]:
            self.model = self.build()
        if FLAGS.train_mode in ["both", "classifier"]:
            if model_path:
                # Load the model from saved ".h5" file
                print(f"\nloading model ...\n")
                print(model_path)
                self.classifier = KM.load_model(
                    filepath=model_path,
                    custom_objects={
                        "mlti": multi_layer_focal(),
                        "MultiLayerAccuracy": MultiLayerAccuracy(),
                    },
                    compile=True,
                )
                # set epoch number
                self.set_epoch(model_path)
            else:
                self.classifier = self.build_classify(num_classes=10)
                # Initialize the epoch counter for model training
                self.epoch = 0

        if FLAGS.train_mode == "combined":

            if model_path:
                # Load the model from saved ".h5" file
                print(f"\nloading model ...\n")
                print(model_path)
                self.combined = KM.load_model(
                    filepath=model_path,
                    custom_objects={
                        "DropBlock2D": DropBlock2D,
                        "mlti": multi_layer_focal(),
                        "MultiLayerAccuracy": MultiLayerAccuracy(),
                    },
                    compile=True,
                )
                # set epoch number
                self.set_epoch(model_path)
            else:
                self.combined = self.build_combined(num_classes=10)

                # Initialize the epoch counter for model training
                self.epoch = 0

    def set_epoch(self, path: str) -> None:
        split = re.split(r"/", path)
        e = [a for a in split if "model_" in a][0].replace(".h5", "")
        e = re.findall(r"[-+]?\d*\.\d+|\d+", e)
        self.epoch = int(e[0])

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
            padding="same",
            use_bias=False,
            name=name + f"_conv_{1}",
        )(input_tensor)
        encoded = KL.Activation("relu")(KL.BatchNormalization()(encoded))
        encoded_list = [encoded]

        skip_tensors = [
            KL.AveragePooling2D(pool_size=(2, 2), strides=2)(
                KL.Activation("relu")(
                    KL.BatchNormalization()(
                        KL.Conv2D(features[0], 1, strides=1, use_bias=False)(
                            input_tensor
                        )
                    )
                )
            ),
            encoded,
        ]
        for i, feature_num in enumerate(features[1:], start=2):
            encoded, skip_tensors = conv_block(
                encoded,
                skip_tensors,
                features=feature_num,
                name=name + f"_conv_{i}",
            )
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

        # For decoder number of features in opposite order of encoder
        decoder_features = self.encoder_features.copy()
        decoder_features.reverse()

        # build the encoder model
        self.encoder_model = self.encoder(
            features=self.encoder_features, name="encoder"
        )

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

        # build the encoder model
        self.encoder_model = self.encoder(
            features=self.encoder_features, name="encoder"
        )
        encoded_features = self.encoder_model(input_tensor)
        contrastive_features = KL.Lambda(lambda x: K.mean(x, [1, 2]), name="contrast")(
            encoded_features[-1]
        )
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
        return KM.Model(
            inputs=input_tensor,
            outputs=[probs, contrastive_features],
            name="classifier",
        )

    def build_combined(self, num_classes: int = 10) -> KM.Model:
        """Method to build combined AE-classifier model

        Returns:
            KM.Model: AE-Classifier model
        """

        # For decoder number of features in opposite order of encoder
        decoder_features = self.encoder_features.copy()
        decoder_features.reverse()

        # build the encoder model
        self.encoder_model = self.encoder(
            features=self.encoder_features, name="encoder"
        )

        # build the decoder model
        decoder = self.decoder(features=decoder_features, name="decoder")
        input_tensor = KL.Input(
            shape=(32, 32, 3)
        )  # shape of images for cifar10 dataset

        # Encode the images
        encoded_features = self.encoder_model(input_tensor)
        # Decode the image from the final layer features of Auto-encoder
        decoded = decoder(encoded_features[-1])
        contrastive_features = KL.Lambda(lambda x: K.mean(x, [1, 2]), name="contrast")(
            encoded_features[-1]
        )

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

        return KM.Model(
            inputs=input_tensor,
            outputs=[decoded, probs, contrastive_features],
            name="combined",
        )

    def compile(self, classifier_loss: str = "focal"):
        """method to compile the model object with optimizer, loss definitions and metrics

        Args:
            classifier_loss (str, optional): loss function to use for classifier training. Defaults to "focal".

        """
        c_loss = contrastive_loss(hidden_norm=True, temperature=0.5, weights=1.0)
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
        focal = multi_layer_focal(gamma=FLAGS.gamma)
        accuracy = MultiLayerAccuracy()
        if FLAGS.train_mode in ["both", "classifier"]:
            self.classifier.compile(
                optimizer=tf.keras.optimizers.Adam(
                    # Linear lr scaling, with batch size (default = 32)
                    learning_rate=FLAGS.lr
                    * (FLAGS.train_batch_size / 32)
                ),
                loss={
                    "logits": focal if classifier_loss == "focal" else cce,
                    "contrast": c_loss,
                },
                metrics={"logits": accuracy, "contrast": None},
                loss_weights={"logits": 10.0, "contrast": 0.0},
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
                    "contrast": c_loss,
                },
                metrics={"decoder": None, "logits": accuracy, "contrast": None},
                loss_weights={"decoder": 1.0, "logits": 10.0, "contrast": 0.0},
            )

    def callbacks(self, val_generator: DataGenerator) -> List[Callback]:
        """method to define all the required callbacks

        Args:
            val_generator (DataGenerator): validation dataset generator function

        Returns:
            List[Callback]: list of all the callbacks
        """
        if FLAGS.train_mode == "classifier":
            model = self.classifier
            model_dir = "class_model"
        elif FLAGS.train_mode == "combined":
            model = self.combined
            model_dir = "com_model"

        # Callback for evaluating the validation dataset
        eval_callback = EvalCallback(model=model, val_generator=val_generator)

        # callback for saving the best model
        checkpoint_callback = ModelCheckpoint(
            f"{model_dir}/{model_dir}" + "_{epoch:04d}_{val_acc:.4f}.h5",
            monitor="val_acc",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="max",
        )

        "Make sure checkpoint callback is after the eval_callback, dependency"
        return [eval_callback, checkpoint_callback]

    def train(self, epochs: int = 10, classifier_loss: str = "focal"):
        """method to initiate model training

        Args:
            epochs (int, optional): total number of training epochs. Defaults to 10.
            classifier_loss(str, optional): loss function for classifier training. Defaults to "focal
        """

        # compile the model object
        if not self.model_path:
            self.compile(classifier_loss=classifier_loss)

        # Common validation dataset format across all training regimes
        val_generator = DataGenerator(
            batch_size=FLAGS.val_batch_size,
            split="val",
            cache=FLAGS.cache,
            train_mode="classifier",
        )
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
            # Directory for saving the trained model
            os.makedirs("./class_model", exist_ok=True)

            # prepare the generators for classifier training
            train_generator_classifier = DataGenerator(
                batch_size=FLAGS.train_batch_size,
                split="train",
                augment=True,
                contrastive=True,
                cache=FLAGS.cache,
                shuffle=True,
                train_mode="classifier",
            )

            # number of trainig steps per epoch
            train_steps = len(train_generator_classifier)

            # if FLAGS.train_mode == "both":
            #     # Use feature representations learnt from AutoEncoder training
            #     self.encoder_model.trainable = True

            self.classifier.fit(
                train_generator_classifier(),
                initial_epoch=self.epoch,
                epochs=epochs,
                workers=8,
                verbose=2,
                steps_per_epoch=train_steps,
                callbacks=self.callbacks(val_generator),
            )

        if FLAGS.train_mode == "combined":
            # Directory for saving the trained model
            os.makedirs("./com_model", exist_ok=True)

            # prepare the generators for classifier training
            train_generator = DataGenerator(
                batch_size=FLAGS.train_batch_size,
                split="train",
                augment=True,
                contrastive=True,
                shuffle=True,
                cache=FLAGS.cache,
                train_mode="combined",
            )

            # number of trainig steps per epoch
            train_steps = len(train_generator)

            self.combined.fit(
                train_generator(),
                initial_epoch=self.epoch,
                epochs=epochs,
                workers=8,
                verbose=2,
                steps_per_epoch=train_steps,
                callbacks=self.callbacks(val_generator),
            )

    def eval(self):
        val_generator = DataGenerator(
            batch_size=FLAGS.val_batch_size,
            split="val",
            cache=FLAGS.cache,
            train_mode="classifier",
        )
        if FLAGS.train_mode == "combined":
            model = KM.Model(
                inputs=self.combined.input,
                outputs=self.combined.get_layer("logits").output,
            )
        elif FLAGS.train_mode == "classifier":
            model = KM.Model(
                inputs=self.classifier.input,
                outputs=self.classifier.get_layer("logits").output,
            )

        # initialize the array to store preds for each label
        accuracy = np.zeros((10, 10), dtype=int)

        for input, true_logits in val_generator():
            pred_logits = model.predict(input)

            true_logits = tf.split(true_logits, num_or_size_splits=4, axis=-1)
            true_logits = true_logits[0]

            # Split the logits from different levels
            pred_logits = tf.split(
                tf.expand_dims(pred_logits, axis=-1), num_or_size_splits=4, axis=1
            )
            # Predicted label by taking an elementwise maximum across all layers
            pred_logits = tf.reduce_max(tf.concat(pred_logits, axis=2), axis=2)

            # Get true and pred labels
            true_labels = tf.argmax(true_logits, axis=-1)
            pred_labels = tf.argmax(pred_logits, axis=-1)
            for i, gt_label in enumerate(true_labels):
                pred_label = int(pred_labels[i])
                accuracy[int(gt_label)][pred_label] += 1

        import matplotlib.pyplot as plt
        import seaborn as sn

        plt.figure(figsize=(10, 7))
        sn.heatmap(accuracy / np.sum(accuracy, axis=-1), annot=True)
        plt.show()
        # metrics = self.combined.evaluate(
        #     val_generator(),
        # )
        # print(metrics)


if __name__ == "__main__":
    model = Cifar()
    model.train()
