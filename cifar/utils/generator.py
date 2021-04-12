import tensorflow_datasets as tfds


class DataGenerator:
    def __init__(self) -> None:
        ds, ds_info = tfds.load("cifar10", split="train", with_info=True)


if __name__ == "__main__":
    generator = DataGenerator()
