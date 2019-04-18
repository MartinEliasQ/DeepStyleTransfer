# from dts import load_images
import sys
import dst.tools as tools
import tensorflow as tf
from tensorflow import keras
K = keras.backend


class dst(object):
    ''' dts class: Deep Style Transfer.

        Args:


        Returns:

    '''

    def __init__(self, *args):
        tools._get_version()

    @staticmethod
    def _get_vgg19():
        ''' Get pre-trained(ImageNet) VGG19 Model without classifier.
            Return:
                  vgg19 model(Keras)
        '''
        vgg19 = keras.applications.vgg19.VGG19(
            weights='imagenet',  include_top=False)
        return vgg19

    @staticmethod
    def _freeze_layers(model):
        """Freeze layers in the model
           Args:
                model: Model
        """
        for layer in model.layers:
            layer.trainable = False
        return model

    @staticmethod
    def _get_dict_layers(model):
        ''' Get Layers in the dict in format {name:layer}
            Args:
                model: Model
            Return:
                  Dictionary with the layers and names.
        '''
        layers = {layer.name: layer.output for layer in model.layers}
        return layers

    @staticmethod
    def _get_layers(model, layers):
        ''' Get layers specify in layers
            Args:
                model: Model
                layers: Array with the name of the layers
            Return:
                   Array with the specific layers
        '''
        return [model.get_layer(layer).output for layer in layers]

    @staticmethod
    def _gram_matrix(feacture_map):
        ''' Compute gram matrix to get feactures of feacture map.
            Reference: http://mathworld.wolfram.com/GramMatrix.html
            Args:
                feacture_map: Each convolutional Layer
            Return:
                  Dot product between flatten feacture map and
                  the respective traspose
        '''
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
    def transfer_style(_content, _style):
        content, style = tools.load_images(_content, _style)
