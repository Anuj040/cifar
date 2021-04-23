import os
import re
import sys
from typing import List, Tuple

sys.path.append("./")
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from keras_drop_block import DropBlock2D
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint

from cifar.utils.callbacks import EvalCallback
from cifar.utils.generator import DataGenerator
from cifar.utils.losses import MultiLayerAccuracy, contrastive_loss, multi_layer_focal


def deconv_block(input_tensor: tf.Tensor, features: int, name: str) -> tf.Tensor:
    """Bottleneck style deconvolutional block. Applies convolution to decrease to fewer features.
    upscales in lower feature space. Convolves to specified feature numbers

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
    input_tensor: tf.Tensor,
    skip_tensors: List[tf.Tensor],
    features_in: int,
    features_out: int,
    name: str,
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """Bottleneck style convolutional block with skip connections across blocks. Applies downsample convolution in lower features space.
    Followed by single kernel convolution to higher feature space. Also, applies skip connections across blocks for routing information
    across levels.

    Args:
        input_tensor (tf.Tensor): tensor to apply convolution to
        skip_tensors (List[tf.Tensor]): list of output tensors from previous blocks
        features_in (int): number of features for incoming layer
        features_out (int): number of features for outgoing layer
        name (str): name for the tensors

    Returns:
        Tuple[tf.Tensor, List[tf.Tensor]]: output tensor, list of tensors to route info to next levels
    """

    out = KL.Conv2D(
        features_in,
        1,
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=name + f"_c{1}",
    )(input_tensor)
    out = KL.Activation("relu")(KL.BatchNormalization()(out))

    out = KL.Conv2D(
        features_in,
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name=name + f"_c{2}",
    )(out)
    out = KL.Activation("relu")(KL.BatchNormalization()(out))

    out = KL.Conv2D(
        features_out,
        1,
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=name + f"_c{3}",
    )(out)
    out = KL.Activation("relu")(KL.BatchNormalization()(out))
    out = KL.SpatialDropout2D(rate=0.2)(out)

    # Calculate skip tensor from previous levels
    skip_connection = KL.AveragePooling2D(pool_size=(2, 2), strides=2)(skip_tensors)
    out = KL.Activation("relu", name=name + "_relu")(out + skip_connection)

    # skip tensor for next level
    skip_tensors = tf.concat([skip_connection, out], axis=-1)
    return out, skip_tensors


def classifier_block(
    encoded_features: List[tf.Tensor],
    num_classes: int = 10,
    activation: str = "sigmoid",
) -> List[tf.Tensor]:
    """shared classifier block across different training methods

    Args:
        encoded_features (List[tf.Tensor]): list of latent feature tensors from different levels
        num_classes (int, optional): total classes. Defaults to 10.
        activation (str, optional): activation function for classification layers. Defaults to "sigmoid".

    Returns:
        List[tf.Tensor]: class logits from different levels
    """

    # logits from the final layer of features of auto-encoder
    encoded_flat = KL.Flatten()(encoded_features[-1])
    encoded_flat = KL.Dropout(rate=0.2)(encoded_flat)
    probs = KL.Dense(num_classes, activation=activation, name="logits_n")(encoded_flat)

    # logits from the all but last layer of features of auto-encoder
    pooled_probs = [
        KL.Dense(num_classes, activation=activation, name=f"logits_{i}")(
            KL.Dropout(rate=0.2)(KL.Flatten()(KL.GlobalAveragePooling2D()(features)))
        )
        for i, features in enumerate(encoded_features[:-1], start=1)
    ]

    # concatenated logits from all the layers
    probs = KL.Lambda(lambda x: tf.concat(x, axis=-1), name="logits")(
        [probs] + pooled_probs
    )
    return probs


class Cifar:
    """

    Args:
        model_path (str): path to the saved model to resume training from
        train_mmode (str): Whether to train encoder, classifier in succession or individually or combined
    """

    def __init__(self, model_path: str, train_mode: str = "combined") -> None:

        self.model_path = model_path
        self.train_mode = train_mode

        # Number of features in successive hidden layers of encoder
        self.encoder_features = [128, 256, 512, 1024]

        if self.train_mode in ["both", "pretrain"]:
            self.model = self.build()

            # Initialize the epoch counter for model training
            self.epoch = 0

        if self.train_mode in ["both", "classifier"]:
            if model_path:
                # Load the model from saved ".h5" file
                print(f"\nloading model ...\n{model_path}\n")
                self.classifier = KM.load_model(
                    filepath=model_path,
                    custom_objects={
                        "mlti": multi_layer_focal(),
                        "MultiLayerAccuracy": MultiLayerAccuracy(),
                        "con": contrastive_loss(),
                    },
                    compile=True,
                )
                # Number of encoder feature levels
                self.n_blocks = len(self.classifier.get_layer("encoder").output)

                # set epoch number
                self.set_epoch(model_path)
            else:
                self.classifier = self.build_classify(num_classes=10)
                # Number of encoder feature levels
                self.n_blocks = len(self.encoder_features)

                # Initialize the epoch counter for model training
                self.epoch = 0

        if self.train_mode == "combined":

            if model_path:
                # Load the model from saved ".h5" file
                print(f"\nloading model ...\n{model_path}\n")
                self.combined = KM.load_model(
                    filepath=model_path,
                    custom_objects={
                        "DropBlock2D": DropBlock2D,
                        "mlti": multi_layer_focal(),
                        "MultiLayerAccuracy": MultiLayerAccuracy(),
                        "con": contrastive_loss(),
                    },
                    compile=True,
                )
                # Number of encoder feature levels
                self.n_blocks = len(self.combined.get_layer("encoder").output)

                # set epoch number
                self.set_epoch(model_path)
            else:
                self.combined = self.build_combined(num_classes=10)
                # Number of encoder feature levels
                self.n_blocks = len(self.encoder_features)

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

        # Prepare the skip tensor from input
        skip_input_tensor = KL.Activation("relu")(
            KL.BatchNormalization()(
                KL.Conv2D(features[0], 1, strides=1, use_bias=False)(input_tensor)
            )
        )
        skip_input_tensor = KL.SpatialDropout2D(rate=0.2)(skip_input_tensor)
        skip_input_tensor = KL.AveragePooling2D(pool_size=(2, 2), strides=2)(
            skip_input_tensor
        )
        skip_tensors = tf.concat(
            [
                skip_input_tensor,  # Routing info from input tensor to next levels
                encoded,  # Routes info from second level to next levels
            ],
            axis=-1,
        )
        for i, feature_num in enumerate(features[1:], start=2):
            encoded, skip_tensors = conv_block(
                encoded,
                skip_tensors,
                features_in=features[i - 2],
                features_out=feature_num,
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
        input_tensor = KL.Input(shape=(2, 2, features[0]))

        decoded = input_tensor

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
        # if self.train_mode == "classifier":
        #     # Use the pretrained encoder for classifier only training
        #     self.encoder_model = KM.load_model("ae_model/ae_model.h5").get_layer(
        #         "encoder"
        #     )
        if self.train_mode != "both":
            # build the encoder model
            self.encoder_model = self.encoder(
                features=self.encoder_features, name="encoder"
            )
        encoded_features = self.encoder_model(input_tensor)
        contrastive_features = KL.Lambda(lambda x: K.mean(x, [1, 2]), name="contrast")(
            encoded_features[-1]
        )
        # Calculate class probs from multiple latent representations
        probs = classifier_block(encoded_features, num_classes=num_classes)

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
        # Calculate class probs from multiple latent representations
        probs = classifier_block(encoded_features, num_classes=num_classes)

        return KM.Model(
            inputs=input_tensor,
            outputs=[decoded, probs, contrastive_features],
            name="combined",
        )

    def compile(
        self,
        classifier_loss: str = "focal",
        lr: float = 1e-3,
        gamma: float = 2.0,
        train_batch_size: int = 32,
    ):
        """method to compile the model object with optimizer, loss definitions and metrics

        Args:
            classifier_loss (str, optional): loss function to use for classifier training. Defaults to "focal".
            lr (float, optional): learning rate for training. defaults to 1e-3
            gamma (float, optional): focussing parameter for focal loss. defaults to 2.0
            train_batch_size (int, optional): batchsize for train dataset. Defaults to 32.
        """
        c_loss = contrastive_loss(hidden_norm=True, temperature=0.5, weights=1.0)
        if self.train_mode in ["both", "pretrain"]:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    # Linear lr scaling, with batch size (default = 32)
                    learning_rate=lr
                    * (train_batch_size / 32)
                ),
                loss="mse",
                metrics=None,
                loss_weights=0.1,
            )
        cce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        )
        focal = multi_layer_focal(gamma=gamma, layers=self.n_blocks)
        accuracy = MultiLayerAccuracy(layers=self.n_blocks)
        if self.train_mode in ["both", "classifier"]:
            self.classifier.compile(
                optimizer=tf.keras.optimizers.Adam(
                    # Linear lr scaling, with batch size (default = 32)
                    learning_rate=lr
                    * (train_batch_size / 32)
                ),
                loss={
                    "logits": focal if classifier_loss == "focal" else cce,
                    "contrast": c_loss,
                },
                metrics={"logits": accuracy, "contrast": None},
                loss_weights={"logits": 10.0, "contrast": 0.0},
            )
        if self.train_mode == "combined":
            self.combined.compile(
                optimizer=tf.keras.optimizers.Adam(
                    # Linear lr scaling, with batch size (default = 32)
                    learning_rate=lr
                    * (train_batch_size / 32)
                ),
                loss={
                    "decoder": "mse",
                    "logits": focal if classifier_loss == "focal" else cce,
                    "contrast": c_loss,
                },
                metrics={"decoder": None, "logits": accuracy, "contrast": None},
                loss_weights={"decoder": 0.1, "logits": 10.0, "contrast": 0.0},
            )

    def callbacks(self, val_generator: DataGenerator) -> List[Callback]:
        """method to define all the required callbacks

        Args:
            val_generator (DataGenerator): validation dataset generator function

        Returns:
            List[Callback]: list of all the callbacks
        """
        if self.train_mode in ["classifier", "both"]:
            model = self.classifier
            model_dir = "class_model"
        elif self.train_mode == "combined":
            model = self.combined
            model_dir = "com_model"

        # Callback for evaluating the validation dataset
        eval_callback = EvalCallback(
            model=model, val_generator=val_generator, layers=self.n_blocks
        )

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

    def train(
        self,
        epochs: int = 10,
        train_steps: int = None,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        classifier_loss: str = "focal",
        lr: float = 1e-3,
        gamma: float = 2.0,
        cache: bool = False,
    ):
        """method to initiate model training

        Args:
            epochs (int, optional): total number of training epochs. Defaults to 10.
            train_batch_size (int, optional): batchsize for train dataset. Defaults to 32.
            val_batch_size (int, optional): batchsize for validation dataset. Defaults to 32.
            classifier_loss (str, optional): loss function for classifier training. Defaults to "focal
            lr (float, optional): learning rate for training. defaults to 1e-3
            gamma (float, optional): focussing parameter for focal loss. defaults to 2.0
            cache (bool, optional): whether to store the train/val data in cache. defaults to False
        """

        # compile the model object
        if not self.model_path:
            self.compile(
                classifier_loss=classifier_loss,
                lr=lr,
                gamma=gamma,
                train_batch_size=train_batch_size,
            )

        # Common validation dataset format across all training regimes
        val_generator = DataGenerator(
            batch_size=val_batch_size,
            split="val",
            layers=self.n_blocks,
            cache=cache,
            train_mode="classifier",
        )
        if self.train_mode in ["both", "pretrain"]:
            # prepare the generator
            train_generator = DataGenerator(
                batch_size=train_batch_size,
                split="train",
                augment=True,
                shuffle=True,
                cache=cache,
                train_mode="pretrain",
            )
            # number of trainig steps per epoch
            train_steps = len(train_generator) if train_steps is None else train_steps
            self.model.fit(
                train_generator(),
                initial_epoch=self.epoch,
                epochs=epochs,
                workers=8,
                verbose=2,
                steps_per_epoch=train_steps,
            )

            # Save the trained autoencoder model
            os.makedirs("./ae_model", exist_ok=True)
            self.model.save("ae_model/ae_model.h5")

            if self.train_mode == "both":
                self.epoch = 0

        if self.train_mode in ["both", "classifier"]:
            # Directory for saving the trained model
            os.makedirs("./class_model", exist_ok=True)

            # prepare the generators for classifier training
            train_generator_classifier = DataGenerator(
                batch_size=train_batch_size,
                split="train",
                layers=self.n_blocks,
                augment=True,
                contrastive=True,
                cache=cache,
                shuffle=True,
                train_mode="classifier",
            )

            # number of trainig steps per epoch
            train_steps = (
                len(train_generator_classifier) if train_steps is None else train_steps
            )

            # if self.train_mode == "both":
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

        if self.train_mode == "combined":
            # Directory for saving the trained model
            os.makedirs("./com_model", exist_ok=True)

            # prepare the generators for classifier training
            train_generator = DataGenerator(
                batch_size=train_batch_size,
                split="train",
                layers=self.n_blocks,
                augment=True,
                contrastive=True,
                shuffle=True,
                cache=cache,
                train_mode="combined",
            )

            # number of trainig steps per epoch
            train_steps = len(train_generator) if train_steps is None else train_steps

            self.combined.fit(
                train_generator(),
                initial_epoch=self.epoch,
                epochs=epochs,
                workers=8,
                verbose=2,
                steps_per_epoch=train_steps,
                callbacks=self.callbacks(val_generator),
            )
            # os.makedirs("./com_model", exist_ok=True)
            # self.combined.save("com_model/com_model.h5")

    def eval(self, val_batch_size: int = 32):
        """Method to run evaluation on the given dataset

        Args:
            val_batch_size (int, optional): batchsize for validation dataset. Defaults to 32.
        """

        val_generator = DataGenerator(
            batch_size=val_batch_size,
            split="test",
            layers=self.n_blocks,
            train_mode="classifier",
        )
        if self.train_mode == "combined":
            model = KM.Model(
                inputs=self.combined.input,
                outputs=self.combined.get_layer("logits").output,
            )
        elif self.train_mode == "classifier":
            model = KM.Model(
                inputs=self.classifier.input,
                outputs=self.classifier.get_layer("logits").output,
            )

        # initialize the array to store preds for each label
        accuracy = np.zeros((10, 10), dtype=int)

        for input, true_logits in val_generator():
            pred_logits = model.predict(input)

            true_logits = tf.split(
                true_logits, num_or_size_splits=self.n_blocks, axis=-1
            )
            true_logits = true_logits[0]

            # Split the logits from different levels
            pred_logits = tf.split(
                tf.expand_dims(pred_logits, axis=-1),
                num_or_size_splits=self.n_blocks,
                axis=1,
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

    def infer(self, image_path: str = None):
        """Method to run single image inference

        Args:
            image_path (str, optional): path to the image to be inferred. Defaults to None.
        """
        # Class labels
        labels = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        # Retrieve the image
        img = np.array(
            tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb")
        )
        # Convert it into model suitable form
        img = 2.0 * tf.cast(tf.expand_dims(img, axis=0), tf.float32) / 255.0 - 1.0

        # Prepare the model object
        if self.train_mode == "combined":
            model = KM.Model(
                inputs=self.combined.input,
                outputs=self.combined.get_layer("logits").output,
            )
        elif self.train_mode == "classifier":
            model = KM.Model(
                inputs=self.classifier.input,
                outputs=self.classifier.get_layer("logits").output,
            )

        # Run through the model
        pred_logits = model.predict(img)

        # Split the logits from different levels
        pred_logits = tf.split(
            tf.expand_dims(pred_logits, axis=-1),
            num_or_size_splits=self.n_blocks,
            axis=1,
        )
        # Predicted label by taking an elementwise maximum across all layers
        pred_logits = tf.reduce_max(tf.concat(pred_logits, axis=2), axis=2)
        # Get pred labels
        pred_labels = tf.argmax(pred_logits, axis=-1)
        label = labels[pred_labels[0]]
        upper = "_" * (31 + len(label))
        lower = "-" * (31 + len(label))
        print(f"{upper}\nThis image belongs to '{label}' class.\n{lower}")


if __name__ == "__main__":
    model = Cifar()
    model.train()
