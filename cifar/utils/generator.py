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


if __name__ == "__main__":
    generator = DataGenerator()
