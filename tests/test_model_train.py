import numpy as np
import tensorflow.keras.models as KM
from cifar.model import Cifar


def test_model_instances():
    """test to check the returned object types of different training configurations"""
    model = Cifar(None, train_mode="both")
    assert isinstance(model.model, KM.Model)
    assert isinstance(model.classifier, KM.Model)

    model = Cifar(None, train_mode="combined")
    assert isinstance(model.combined, KM.Model)


def test_weight_change_both():
    """test to check the model training is changing the trainable weights"""

    # get the model defining object
    model = Cifar(None, train_mode="both")

    # get the weights before training for a convolutional layer in encoder of Autoencoder model
    weights_before_train = (
        model.model.get_layer("encoder").get_layer("encoder_conv_1").get_weights()[0]
    )
    # get the weights before training for final dense layer of classifier model
    weights_before_train_1 = model.classifier.get_layer("logits_n").get_weights()[0]
    model.train(epochs=2, train_steps=2)
    # get the weights after training for a convolutional layer in encoder of Autoencoder model
    weights_after_train = (
        model.model.get_layer("encoder").get_layer("encoder_conv_1").get_weights()[0]
    )
    # get the weights after training for final dense layer of classifier model
    weights_after_train_1 = model.classifier.get_layer("logits_n").get_weights()[0]

    assert not np.all(weights_after_train == weights_before_train)
    assert not np.all(weights_after_train_1 == weights_before_train_1)


def test_weight_frozen_both():
    """test to ensure that freezing the layer keeps the weights unchanged despite model training"""

    # get the model defining object
    model = Cifar(None, train_mode="both")

    # Freeze the layers
    model.model.get_layer("encoder").trainable = False
    model.classifier.get_layer("logits_n").trainable = False

    # get the weights before training for a convolutional layer in encoder of Autoencoder model
    weights_before_train = (
        model.model.get_layer("encoder").get_layer("encoder_conv_1").get_weights()[0]
    )
    # get the weights before training for final dense layer of classifier model
    weights_before_train_1 = model.classifier.get_layer("logits_n").get_weights()[0]
    model.train(epochs=2, train_steps=2)
    # get the weights after training for a convolutional layer in encoder of Autoencoder model
    weights_after_train = (
        model.model.get_layer("encoder").get_layer("encoder_conv_1").get_weights()[0]
    )
    # get the weights after training for final dense layer of classifier model
    weights_after_train_1 = model.classifier.get_layer("logits_n").get_weights()[0]

    assert np.all(weights_after_train == weights_before_train)
    assert np.all(weights_after_train_1 == weights_before_train_1)


def test_model_eval_infer_coverage():
    model = Cifar(None, train_mode="classifier")
    model.eval()
    model.infer("./test0.png")
