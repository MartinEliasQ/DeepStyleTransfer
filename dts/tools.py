"""Generic tools"""
import matplotlib.pyplot as plt


def _load_image(path_image: str = ""):
    """ Load image
        path_image: Path of the image
        return-> numpy array with the image"""
    return plt.imread(path_image)


def load_images(path_image1, path_image2):
    """Load a couple image and return tuple with
        two numpy arrays(each image)"""
    return (_load_image(path_image1), _load_image(path_image2))
