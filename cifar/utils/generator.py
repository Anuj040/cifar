import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds


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


class DataGenerator:
    def __init__(self, split: str = "train") -> None:

        # Retrieve the dataset
        ds, ds_info = tfds.load("cifar10", split=split, with_info=True)

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
        if split == "train":
            # Labels with fewer training samples (imbalanced labels)
            screened_labels = ["bird", "deer", "truck"]
            # Samples to be selected
            imbalance_sample_size = 2500
            # retrieve indices for imbalanced labels in the labels list
            screened_labels_id = [labels.index(label) for label in screened_labels]
            # List of image ids to be used for training for the imabalanced labels
            allowed_ids = allowed_id_list(ds, screened_labels_id, imbalance_sample_size)


if __name__ == "__main__":
    generator = DataGenerator()
