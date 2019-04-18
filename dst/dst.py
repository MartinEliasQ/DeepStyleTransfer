# from dts import load_images
import sys
import dst.tools as tools
import tensorflow as tf
from tensorflow import keras
K = keras.backend


class dst(object):
    """
    Deep  Style Transfer Class
    """

    def __init__(self, *args):
        dst._get_version()
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
    def _get_layers(model):
        layers = {layer.name: layer.output for layer in model.layers}
        return layers

    @staticmethod
    def _gram_matrix(feacture_map):
        """"""
        return K.dot(feacture_map, K.transpose(feacture_map))

    @staticmethod
    def _content_loss(content, generated):
        return K.sum(K.square(content, generated))

    @staticmethod
    def _style_loss(style, generated):
        pass

    @staticmethod
    def total_loss(content_loss, style_loss):
        return content_loss + style_loss

    @staticmethod
    def _get_version():
        print("Python version: {}.{}".format(
            sys.version_info[0], sys.version_info[1]))
        print('TensoFlow version: ', tf.__version__)
        print('Keras version: ', keras.__version__)

    @staticmethod
    def transfer_style(_content, _style):
        content, style = tools.load_images(_content, _style)

    @staticmethod
    def _select_layer(parameter_list):
        raise NotImplementedError
