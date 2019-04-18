"""Generic tools"""
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from PIL import Image


def _open_image(path_image: str = ""):
    """ Load image
        path_image: Path of the image
        return-> numpy array with the image"""
    return Image.open(path_image)


def _image_to_array(image):
    return np.array(image)


def _add_dim(tensor):
    return np.expand_dims(tensor, axis=0)


def _del_dim(tensor):
    return np.squeeze(tensor, axis=0)


def _scale_image(image, max_dim=512):
    # ANTIALIAS -> (a high-quality downsampling filter).
    scale = max_dim / max(image.size)
    return image.resize((round(image.size[0]*scale),
                         round(image.size[1]*scale)), Image.ANTIALIAS)


def get_image_dim(image):
    return image.shape


def _load_image(path_image, max_dim=512):
    img = _open_image(path_image)
    img = _scale_image(img, max_dim)
    img = _image_to_array(img)
    img = _add_dim(img)
    return img.astype('uint8')


def imshow(image, title=None):
    # Remove the batch dimension
    out = np.squeeze(image, axis=0)
    # Normalize for display
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
        plt.imshow(out)


def _load_and_process_img(path_image, max_dim=512):
    img = _load_image(path_image, max_dim)

    # Preprocess input image to vgg requierements
    # That is a normalize(central) pixel of image
    return keras.applications.vgg19.preprocess_input(img)
