"""Generic tools"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import requests
from io import BytesIO
import sys


def _open_image(image_path: str = None, url=False):
    ''' Load the image that is in the  image_path

        Args:
             path_image(str): Source address where the image is located
             url(bool): image_path is url

        Returns:
                image: Image
    '''
    assert(image_path is not None), "You should provide an image path"
    if url:
        response = requests.get(image_path)
        image_path = BytesIO(response.content)

    return Image.open(image_path)


def _image_to_array(image):
    ''' Convert image object to numpy array(generally a matrix with 3 dimensions)

        Args:
             image (Image object): image to convert to matrix
        Returns:
                image: Matrix with the values of the image.(numpy.ndarray)
    '''

    return np.array(image)


def _add_dim(tensor):
    ''' Add dimensions of a tensor in the axis 0.

        Args:
             tensor (numpy.ndarray): Tensor
        Returns:
                tensor: Tensor with one more dimension
    '''
    assert(type(tensor) is np.ndarray), "Tensor should be numpy.ndarray"
    return np.expand_dims(tensor, axis=0)


def _del_dim(tensor):
    ''' Delete dimensions of a tensor in the axis 0.

        Args:
             tensor (numpy.ndarray): Tensor
        Returns:
                dims: Tensor.
    '''
    assert(type(tensor) is np.ndarray), "Tensor should be numpy.ndarray"
    return np.squeeze(tensor, axis=0)


def _scale_image(image, max_dim=512):
    ''' The image is scaled relative to the maximum dimension

        Args:
             image (Image Object): Image
             max_dim: Maximum dimension of the image
        Returns:
                Image Object: A image resize with the maxium dimension
    '''

    scale = max_dim / max(image.size)

    #  Image.ANTIALIAS : A Anti-aliasing  is applied
    #                    to the image to make it look better.
    #  ANTIALIAS -> (a high-quality downsampling filter).

    return image.resize((round(image.size[0]*scale),
                         round(image.size[1]*scale)), Image.ANTIALIAS)


def _load_image(path_image, max_dim=512, url=False):
    print()
    img = _open_image(path_image, url=url)
    img = _scale_image(img, max_dim)
    img = _image_to_array(img)
    img = _add_dim(img)
    return img.astype('uint8')


def imshow(img, title=None):
    ''' Show image and title.
        Args:
             img: Image to Show
             title: Title Head of image
    '''

    if len(img.shape) == 4:
        # Remove the batch dimension
        img = np.squeeze(img, axis=0)
    # Normalize for display
    out = img.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)


def _load_and_process_img(path_image, max_dim=512, url=False):
    ''' Load image(open, scale, convert in numpy array and add batch dim to
         image) and preprocess image in the VGG format. (Color format BGR and 
         normalize pixels with the mean)

        Arg:
            path_image(str): Source address where the image is located
            max_dim: Maximun dimension of the image(Scale Relative)
        Returns:
                image preprocess(VGG format)
    '''
    img = _load_image(path_image, max_dim, url=url)

    # Preprocess input image to vgg requierements
    # That is a normalize(central) pixel of image
    img = keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(processed_img, color_format="rgb"):
    ''' Convert a processed image (VGG Format) to normal image(RGB and Non
        Normalize pixels)

        Arg:
            processed_img: Matrix or Tensor with the values of the retult image
            color_format: rgb or bgr format to the image
        Returns:
                Matrix with the pixels without normalize (optional : RGB)
                [If you don't pass rgb arguments, the channels will be BGR ]
    '''
    x = processed_img.copy()

    if len(x.shape) == 4:
        # Remove the batch dimension
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or"
                               "[height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    # That is a inverse step that process image
    # That values are  mean pixel values(Normalize)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # Convert image BGR to RGB - invert colors
    # The data has 3 dimensions: width, height, and color.
    #  ::-1 effectively reverses the order of the colors.
    # The width and height are not affected.
    # Source:
    # https://stackoverflow.com/questions/4661557/pil-rotate-image-colors-bgr-rgb
    # if color_format == 'rgb':
    x = x[:, :, ::-1]
    # Clip Values between 0 and 255 (Pixel Values)
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def _get_version():
    print("Python version: {}.{}".format(
        sys.version_info[0], sys.version_info[1]))
    print('TensoFlow version: ', tf.__version__)
    print('Keras version: ', keras.__version__)


def _get_layers(model, layers):
    ''' Get layers specify in layers vector
        Args:
            model: Model
            layers: Array with the name of the layers
        Return:
                Array with the specific layers
    '''
    return [model.get_layer(layer).output for layer in layers]


def _get_dict_layers(model):
    ''' Get Layers in the dict in format {name:layer}
        Args:
            model: Model
        Return:
                Dictionary with the layers and names.
    '''
    layers = {layer.name: layer.output for layer in model.layers}
    return layers


def _freeze_layers(model):
    """Freeze layers in the model
        Args:
            model: Model
    """
    for layer in model.layers:
        layer.trainable = False
    return model


def generate_image(shape):
    return np.random.rand(shape)*255
