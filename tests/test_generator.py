import numpy as np
from cifar.utils.generator import DataGenerator

# Initializing one instance of train and test generators
train_generator = DataGenerator(shuffle=True)
test_generator = DataGenerator(split="test")

# Retrieving one batch of images and labels for train and test set
first_train_batch, first_train_label = next(iter(train_generator()))
first_test_batch, first_test_label = next(iter(test_generator()))


def test_batch_number():
    """test to make sure that dataset imbalancing leaves the expected number of batches"""

    # original train dataset size = 50000, batch_size = 32
    # Imbalanced: 3 classes reduced by 2500 examples followed by oversampling
    train_length = (50000) // 32

    # test dataset size = 10000, batch_size = 32
    test_length = (10000) // 32 + 1  # last fractional batch also considered
    assert len(train_generator) == train_length
    assert len(test_generator) == test_length


def test_batch_constituents():
    """test to make sure that dataset shuffling is working as expected"""

    # Retrieving second batch of images and labels for train and test set from a new instance of train and test generators
    second_train_batch, second_train_label = next(iter(DataGenerator(shuffle=True)()))
    second_test_batch, second_test_label = next(iter(DataGenerator(split="test")()))

    # Train dataset will be shuffled
    assert not np.all(second_train_batch.numpy() == first_train_batch.numpy())
    assert not np.all(second_train_label.numpy() == first_train_label.numpy())

    # Test dataset not shuffled
    assert np.all(second_test_batch.numpy() == first_test_batch.numpy())
    assert np.all(second_test_label.numpy() == first_test_label.numpy())


def test_batch_pretrain():
    """test to make sure that for feature extractor training datagenerator returns input batch as the output"""

    # Retrieving input and output batch from datagenerator for pretraining of feature extractor
    for input_train_batch, output_train_batch in DataGenerator(
        shuffle=True, train_mode="pretrain"
    )().take(5):

        assert np.all(input_train_batch.numpy() == output_train_batch.numpy())
