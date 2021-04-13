import numpy as np
from cifar.model import Cifar


def test_weight_change():
    """test to check the model training is changing the trainable weights"""

    # get the model defining object
    model = Cifar()

    # get the weights before training for a convolutional layer in encoder of Autoencoder model
    weights_before_train = (
        model.model.get_layer("encoder").get_layer("encoder_conv_2").get_weights()[0]
    )
    # get the weights before training for final dense layer of classifier model
    weights_before_train_1 = model.classifier.get_layer("logits").get_weights()[0]
    model.train(epochs=2)
    # get the weights after training for a convolutional layer in encoder of Autoencoder model
    weights_after_train = (
        model.model.get_layer("encoder").get_layer("encoder_conv_2").get_weights()[0]
    )
    # get the weights after training for final dense layer of classifier model
    weights_after_train_1 = model.classifier.get_layer("logits").get_weights()[0]

    assert not np.all(weights_after_train == weights_before_train)
    assert not np.all(weights_after_train_1 == weights_before_train_1)


def test_weight_frozen():
    """test to ensure that freezing the layer keeps the weights unchanged despite model training"""

    # get the model defining object
    model = Cifar()

    # Freeze the layers
    model.model.get_layer("encoder").trainable = False
    model.classifier.get_layer("logits").trainable = False

    # get the weights before training for a convolutional layer in encoder of Autoencoder model
    weights_before_train = (
        model.model.get_layer("encoder").get_layer("encoder_conv_2").get_weights()[0]
    )
    # get the weights before training for final dense layer of classifier model
    weights_before_train_1 = model.classifier.get_layer("logits").get_weights()[0]
    model.train(epochs=2)
    # get the weights after training for a convolutional layer in encoder of Autoencoder model
    weights_after_train = (
        model.model.get_layer("encoder").get_layer("encoder_conv_2").get_weights()[0]
    )
    # get the weights after training for final dense layer of classifier model
    weights_after_train_1 = model.classifier.get_layer("logits").get_weights()[0]

    assert np.all(weights_after_train == weights_before_train)
    assert np.all(weights_after_train_1 == weights_before_train_1)
