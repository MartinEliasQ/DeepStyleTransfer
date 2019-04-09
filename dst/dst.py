# from dts import load_images
import tensorflow as tf
from tensorflow import keras
K = keras.backend


class dst(object):
    """
    Deep  Style Transfer Class
    """

    def __init__(self, *args):
        pass

    @staticmethod
    def _get_vgg19():
        """Get VGG model and weights with ImageNet"""
        vgg19 = keras.applications.vgg19.VGG19(
            weights='imagenet',  include_top=False)
        return vgg19

    @staticmethod
    def _freeze_layers(model):
        """"""
        for layer in model.layers:
            layer.trainable = False
        return model

    @staticmethod
    def _get_layer(model, layer):
        """"""
        pass

    @staticmethod
    def _gram_matrix(feacture_map):
        """"""
        return K.dot(feacture_map, K.transpose(feacture_map))
