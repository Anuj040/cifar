import functools
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental.preprocessing import (
    RandomRotation,
    RandomTranslation,
)


def allowed_id_list(
    ds: tf.data.Dataset, screened_labels_id: list, imbalance_sample_size: int
) -> np.array:
    """Filterig method for retaining only a subset of image files for training

    Args:
        ds (tf.data.Dataset): dataset iterator. Each element contains id, image, and label
        screened_labels_id (list): indices for which ids will be filtered
        imbalance_sample_size (int): number of ids to retain for screened labels

    Returns:
        np.array: array of ids to be used for training for screened labels
    """
    assert isinstance(ds, tf.data.Dataset)
    # Dictionary for storing ids corresponding to each screened label
    id_list = {id: [] for id in screened_labels_id}
    # Loop over the whole dataset
    for sample in ds:
        id, label = (
            sample["id"].numpy(),
            sample["label"].numpy(),
        )
        # prepare list of ids for given screened label
        if label in screened_labels_id:
            id_list[label].append(id)

    np.random.seed(42)  # For reproducibility

    # List for storing ids to be used for training over all the screened labels
    allowed_ids = []
    for id, sample_list in id_list.items():
        # Retaining only a randomly chosen subset of ids
        allowed_ids.append(
            np.random.choice(sample_list, imbalance_sample_size, replace=False)
        )
    return np.concatenate(allowed_ids)


def random_apply(func, p: float, image: tf.Tensor) -> tf.Tensor:
    """Randomly apply function func to image with probability p."""

    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32),
        ),
        lambda: func(image),
        lambda: image,
    )


def color_jitter_transform(
    image: tf.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> tf.Tensor:
    """Distorts the color of the image (jittering order is random).

    Args:
      image: The input image tensor.
      brightness: A float, specifying the brightness for color jitter.
      contrast: A float, specifying the contrast for color jitter.
      saturation: A float, specifying the saturation for color jitter.
      hue: A float, specifying the hue for color jitter.

    Returns:
      The distorted image tensor.
    """
    with tf.name_scope("distort_color"):

        def apply_transform(i, x):
            """Apply the i-th transformation."""

            def brightness_foo():
                if brightness == 0:
                    return x
                else:
                    return tf.image.random_brightness(x, max_delta=brightness)

            def contrast_foo():
                if contrast == 0:
                    return x
                else:
                    return tf.image.random_contrast(
                        x, lower=1 - contrast, upper=1 + contrast
                    )

            def saturation_foo():
                if saturation == 0:
                    return x
                else:
                    return tf.image.random_saturation(
                        x, lower=1 - saturation, upper=1 + saturation
                    )

            def hue_foo():
                if hue == 0:
                    return x
                else:
                    return tf.image.random_hue(x, max_delta=hue)

            x = tf.cond(
                tf.less(i, 2),
                lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo),
            )
            return x

        perm = tf.random.shuffle(tf.range(4))
        for i in range(4):
            image = apply_transform(perm[i], image)
            image = tf.clip_by_value(image, -1.0, 1.0)
        return image


def color_jitter(image: tf.Tensor, strength: float) -> tf.Tensor:
    """Distorts the color of the image.

    Args:
      image (tf.Tensor): The input image tensor.
      strength (tf.Tensor): the floating number for the strength of the color augmentation.

    Returns:
      The distorted image tensor.
    """
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    return color_jitter_transform(image, brightness, contrast, saturation, hue)


def to_grayscale(image: tf.Tensor) -> tf.Tensor:
    """
    returns a 3-channel grayscale image
    """
    image = tf.image.rgb_to_grayscale(image)
    image = tf.tile(image, [1, 1, 3])
    return image


def image_augmenter(image: tf.Tensor) -> tf.Tensor:
    """Augmenter method working on single images

    Args:
        image (tf.Tensor): single image tensor (shape = HWC)

    Returns:
        tf.Tensor: augmented image tensor
    """

    # Flip LR
    def fliplr(image: tf.Tensor):
        return image[:, ::-1, :]

    # Flip UD
    def flipud(image: tf.Tensor):
        return image[::-1, :, :]

    def channel_shuffle(image: tf.Tensor) -> tf.Tensor:
        """method to randmly shuffle the color channels of the image"""

        # Make the channels axis as the main axis
        image = tf.transpose(image, perm=[2, 1, 0])
        # Shuffle the channels
        image = tf.random.shuffle(image)
        # Return the original axes order
        image = tf.transpose(image, perm=[2, 1, 0])

        return image

    def crop_and_resize(image: tf.Tensor) -> tf.Tensor:
        """takes a random crop from the image and resize it to original image size"""

        crop_proportion = 0.875
        size = (int(32 * crop_proportion), int(32 * crop_proportion), 3)
        image = tf.image.random_crop(image, size)
        image = tf.image.resize(image, (32, 32))

        return image

    def random_color_jitter(image: tf.Tensor, p: float = 1.0) -> tf.Tensor:
        def _transform(image: tf.Tensor) -> tf.Tensor:

            color_jitter_t = functools.partial(color_jitter, strength=0.5)
            image = random_apply(color_jitter_t, p=0.8, image=image)
            return random_apply(to_grayscale, p=0.2, image=image)

        return random_apply(_transform, p=p, image=image)

    image = random_apply(fliplr, p=0.5, image=image)
    image = random_apply(flipud, p=0.5, image=image)
    image = random_apply(crop_and_resize, p=0.5, image=image)
    image = random_apply(channel_shuffle, p=0.0, image=image)
    image = random_apply(random_color_jitter, p=0.5, image=image)

    return image


def sample_beta_distribution(
    size: int, concentration_0: float = 0.2, concentration_1: float = 0.2
) -> tf.Tensor:
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)

    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


class DataGenerator:
    """Train/test dataset generator class

    Args:
        split (str, optional): dataset split to use. Defaults to "train".
        layers (int, optional): number of layers to take classification tensor from. Defaults to 1.
        train_mode (str, optional): Training feature extractor or classifier. Defaults to "pretrain".
        batch_size (int, optional): Defaults to 32.
        augment (bool, optional): whether to augment the images. Defaults to False.
        contrastive (bool, optional): whether to apply contrastive loss for model training. Defaults to False.
        shuffle (bool, optional): whether to shuffle the dataset. Defaults to False.
        cache (bool, optional): dataset will be cached or not. Defaults to False.
        buffer_multiplier (int, optional): Buffer to maintain for faster training. Defaults to 5.

    """

    def __init__(
        self,
        split: str = "train",
        layers: int = 1,
        train_mode: str = "pretrain",
        batch_size: int = 32,
        augment: bool = False,
        contrastive: bool = False,
        shuffle: bool = False,
        cache: bool = False,
        buffer_multiplier: int = 5,
    ) -> None:
        assert split in ["train", "test", "val"]
        self.split = split
        assert train_mode in ["pretrain", "classifier", "combined"]
        self.layers = layers
        self.train_mode = train_mode
        self.batch_size = batch_size
        self.augment = augment
        self.contrast = contrastive

        # Retrieve the dataset
        ds, ds_info = tfds.load(
            "cifar10",
            split="train" if split in ["train", "val"] else split,
            with_info=True,
        )

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
        # Get the number of disctinct labels
        self.num_classes = len(labels)

        # Augmentators
        # Define the rotation & translation augmentator
        self.rotate_translate = tf.keras.Sequential(
            [
                RandomRotation(0.1, fill_mode="constant"),
                RandomTranslation(
                    height_factor=0.1, width_factor=0.1, fill_mode="constant"
                ),
            ]
        )
        if split in ["train", "val"]:
            # Labels with fewer training samples (imbalanced labels)
            screened_labels = ["bird", "deer", "truck"]
            # Samples to be selected
            imbalance_sample_size = 2500
            # retrieve indices for imbalanced labels in the labels list
            screened_labels_id = [labels.index(label) for label in screened_labels]
            # List of image ids to be used for training for the imabalanced labels
            allowed_ids = allowed_id_list(ds, screened_labels_id, imbalance_sample_size)

            def _predicate(
                input: dict,
                check_labels=tf.constant(screened_labels_id),
                allowed_ids=tf.constant(allowed_ids),
            ) -> bool:
                """method to filter the dataset for screened labels and their corresponding id list

                Args:
                    input (dict): an element from the dataset
                    check_labels (tf.constant, optional): labels to be screened. Defaults to tf.constant(screened_labels_id).
                    allowed_ids (tf.constant, optional): retained ids for screened labels. Defaults to tf.constant(allowed_ids).

                Returns:
                    bool: wether to use this particular sample or not
                """
                # Extract image id and label
                id, label = (
                    input["id"],
                    input["label"],
                )
                # check if the label matches one of the screened labels
                screen_labels = tf.equal(
                    check_labels, tf.cast(label, check_labels.dtype)
                )
                # check if the ids match one of the allowed ids for screened labels
                screen_ids = tf.equal(allowed_ids, tf.cast(id, allowed_ids.dtype))

                # check if a match has been detected
                reduced_ids = tf.reduce_sum(tf.cast(screen_ids, tf.float32))
                reduced_labels = tf.reduce_sum(tf.cast(screen_labels, tf.float32))

                return tf.cond(
                    reduced_labels
                    > 0,  # for screened labels id needs to be an allowed one
                    lambda: tf.greater(reduced_ids * reduced_labels, tf.constant(0.0)),
                    lambda: True,  # For non-screened labels
                )

            ds = ds.filter(_predicate)

            def oversample_classes(
                input: dict, check_labels=tf.constant(screened_labels_id)
            ) -> int:
                """
                Returns the number of copies of given sample
                """
                label = input["label"]

                # Check if the label is from underrepresented class
                screen_labels = tf.equal(
                    check_labels, tf.cast(label, check_labels.dtype)
                )
                reduced_labels = tf.reduce_sum(tf.cast(screen_labels, tf.float32))

                # Default number of times an element will be repeated
                repeat_count = tf.constant(1, tf.int64)

                # Repeat the undersampled samples
                residual = tf.less_equal(tf.random.uniform([], dtype=tf.float32), 1.0)
                residual = tf.cast(residual, tf.int64)
                # Oversample samples for label "2" using feedback from validation dataset
                # residual = tf.cond(
                #     tf.equal(label, 2), lambda: residual + 1, lambda: residual
                # )

                return tf.cond(
                    reduced_labels > 0,
                    lambda: repeat_count + residual,
                    lambda: repeat_count,
                )

            if train_mode in ["classifier", "combined"]:
                total_size = len(list(ds))
                if split == "train":
                    ds = ds.take(int(0.8 * total_size))
                elif split == "val":
                    ds = ds.skip(int(0.8 * total_size))
            elif train_mode == "pretrain" and split == "val":
                raise Exception("Train dataset is split only for classifier training")

            # Oversample the under-represented classes once the validation split have been taken
            if split == "train":
                ds = ds.flat_map(
                    lambda x: tf.data.Dataset.from_tensors(x).repeat(
                        oversample_classes(x)
                    )
                )

        if cache:
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(
                batch_size * buffer_multiplier, reshuffle_each_iteration=True
            )
        # Per image mapping
        ds = ds.map(self.map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=split == "train")

        # Per batch mapping
        ds = ds.map(self.batch_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    def map_fn(self, input: dict) -> Tuple[tf.Tensor, tf.Tensor]:
        """method to transform the dataset elements to model usable form

        Args:
            input (dict): an element from the dataset

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: input/output tensor pair
        """
        # Get the image array
        image = 2.0 * tf.cast(input["image"], tf.float32) / 255.0 - 1.0

        # Whether to apply augmentation
        if self.augment:
            if self.contrast:
                image_aug = []
                # Two different augmentated versions for each image
                for _ in range(2):
                    image_aug.append(image_augmenter(image))
                image = tf.concat(image_aug, axis=-1)
            else:
                image = image_augmenter(image)

        if self.split == "train" and self.train_mode == "pretrain":
            return image, image

        return image, tf.one_hot(input["label"], self.num_classes)

    def batch_map_fn(
        self, input_1: tf.Tensor, input_2: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        if self.contrast and self.split == "train":
            # Take the two augmented versions of a single batch concatenated along the channels dimension
            # concatenate them along the batch dimension to make input image batch double the specified batch_size
            input_1 = tf.concat(
                tf.split(input_1, num_or_size_splits=2, axis=-1), axis=0
            )
            input_2 = tf.tile(input_2, [2, 1])

        def _augmenter(
            input_1: tf.Tensor, input_2: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            """method to apply augmentations on a batch of images

            Args:
                input_1 (tf.Tensor): [description]
                input_2 (tf.Tensor): [description]

            Returns:
                Tuple[tf.Tensor, tf.Tensor]: [description]
            """

            def random_rotate_translate(images: tf.Tensor) -> tf.Tensor:
                # randomly rotates each image in the batch
                images = self.rotate_translate(images, training=True)
                return images

            def random_mixup(
                inputs: Tuple[tf.Tensor, tf.Tensor]
            ) -> Tuple[tf.Tensor, tf.Tensor]:

                """method to apply transformations between elements of a given batch.
                Takes a batch and mixes it with its own elements in reverse order.

                Args:
                    input_1, input_2 = inputs[0], inputs[1]
                    input_1 (tf.Tensor): "classifier" & "pretrain": input_image
                    input_2 (tf.Tensor): "classifier": one-hot labels, "pretrain": output_image

                Returns:
                    Tuple[tf.Tensor, tf.Tensor]: input/output tensor pair
                """
                input_1, input_2 = inputs

                # Random mixing proportions for different elements
                if self.contrast:
                    # mixup_proportions = tf.random.uniform(
                    #     shape=[2 * self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0
                    # )
                    # Sample lambda and reshape it to do the mixup
                    l = sample_beta_distribution(2 * self.batch_size, 0.2, 0.2)
                    mixup_proportions = tf.reshape(l, (2 * self.batch_size, 1, 1, 1))
                else:
                    # Sample lambda and reshape it to do the mixup
                    l = sample_beta_distribution(self.batch_size, 0.2, 0.2)
                    mixup_proportions = tf.reshape(l, (self.batch_size, 1, 1, 1))

                input_1 = (
                    mixup_proportions * input_1
                    + (1 - mixup_proportions) * input_1[::-1, ...]
                )
                mixup_proportions = tf.squeeze(mixup_proportions, axis=-1)
                mixup_proportions = tf.squeeze(mixup_proportions, axis=-1)
                input_2 = (
                    mixup_proportions * input_2
                    + (1 - mixup_proportions) * input_2[::-1, ...]
                )
                return input_1, input_2

            input_1 = random_apply(random_rotate_translate, p=0.0, image=input_1)
            input_1, input_2 = random_apply(
                random_mixup, p=0.0, image=(input_1, input_2)
            )

            if self.train_mode not in ["classifier", "combined"]:
                # For pretraining, input and output are the same.
                input_2 = input_1
            else:
                # Classification output for multiple levels
                input_2 = tf.concat([input_2] * self.layers, axis=-1)
            return input_1, input_2

        (input_1, input_2) = tf.cond(
            tf.cast(self.augment, tf.bool),
            lambda: _augmenter(input_1, input_2),
            lambda: (input_1, tf.concat([input_2] * self.layers, axis=-1))
            if self.train_mode in ["classifier", "combined"]
            else (input_1, input_1),  # For pretraining, input and output are the same.
        )
        if self.contrast:
            if self.train_mode == "combined":
                # image, (image, one-hot label * 4, dummy variable)
                return input_1, (input_1, input_2, np.zeros((2 * self.batch_size)))
            return input_1, (input_2, np.zeros((2 * self.batch_size)))
        else:
            if self.train_mode == "combined":
                # image, (image, one-hot label * 4 )
                return input_1, (input_1, input_2)
            return input_1, input_2

    def __call__(self, *args, **kwargs) -> tf.data.Dataset:
        if self.split == "train":
            return self.ds.repeat(-1)  # Batch repition over epochs
        else:
            return self.ds

    def __len__(self) -> int:
        # Get the total number of batches
        return len(list(self.ds))


if __name__ == "__main__":
    train_generator = DataGenerator(
        batch_size=2,
        # shuffle=True,
        augment=True,
        contrastive=True,
        train_mode="combined",
    )
    # test_generator = DataGenerator(split="test")
    # val_generator = DataGenerator(split="val", contrastive=True, train_mode="combined")
    # print(len(test_generator))
    # print(len(train_generator))
    # print(len(val_generator))
    for a, (b, c, d) in train_generator().take(1):
        print(a.shape)
        # print(b)
        print(c.shape)
        print(d)
        tf.keras.preprocessing.image.save_img(f"test1.png", a.numpy()[0], scale=True)
        tf.keras.preprocessing.image.save_img(f"test4.png", a.numpy()[3], scale=True)
