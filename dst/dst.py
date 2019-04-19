# from dts import load_images
import sys
import dst.tools as tools
import numpy as np

import dst.losses as losses
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow import keras
from tensorflow.python.keras import models

import IPython.display

from PIL import Image
import time

K = keras.backend
tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))
CONTENT_LAYERS_LIST = ['block5_conv2']
STYLE_LAYERS_LIST = ['block1_conv1',
                     'block2_conv1',
                     'block3_conv1',
                     'block4_conv1',
                     'block5_conv1'
                     ]


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
    def _get_feactuere_maps(model, content_path, style_path, num_style_layers):
        content_image = tools._load_and_process_img(content_path)
        style_image = tools._load_and_process_img(style_path)
        style_outputs = model(style_image)
        content_outputs = model(content_image)

        style_features = [style_layer[0]
                          for style_layer in style_outputs[:num_style_layers]]
        content_features = [content_layer[0]
                            for content_layer
                            in content_outputs[num_style_layers:]]

        return style_features, content_features

    @staticmethod
    def compute_grads(cfg):
        # Create context to follow gradients
        with tf.GradientTape() as tape:
            all_loss = losses.compute_loss(**cfg)
        # Compute gradients wrt input image
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss

    @staticmethod
    def get_model(content_layers, style_layers):
        vgg = dst._get_vgg19()
        vgg = tools._freeze_layers(vgg)
        style_outputs = [vgg.get_layer(name).output for name in style_layers]
        content_outputs = [vgg.get_layer(
            name).output for name in content_layers]
        model_outputs = style_outputs + content_outputs
        new_model = models.Model(vgg.input, model_outputs)
        return tools._freeze_layers(new_model)

    @staticmethod
    def tranfer_style(content_path, style_path,
                      content_layers=CONTENT_LAYERS_LIST,
                      style_layers=STYLE_LAYERS_LIST,
                      content_weight=1e3, style_weight=1e-2,
                      variation_weight=0,
                      num_iter=1000, init=True):

        # Get New Model with the respective outputs(CNN layers)
        model = dst.get_model(content_layers, style_layers)

        # Get  feacture maps for Style and Content
        style_features, content_features = dst._get_feactuere_maps(
            model, content_path, style_path, len(style_layers))

        # Get Gram Matrix per each Style Layer
        gram_style_features = [losses.gram_matrix(
            style_feature) for style_feature in style_features]

        # Set Init Image
        if init is True:
            init_image = tools._load_and_process_img(content_path)
        else:
            print("Random image Generated")
            partial_image = tools._del_dim(tools._load_image(content_path))
            h, w, c = partial_image.shape
            random_image = tools._add_dim(np.random.rand(h, w, c) * 255)

            init_image = tf.keras.applications.vgg19.preprocess_input(
                random_image)

        # Set in Tensor init image
        init_image = tfe.Variable(init_image, dtype=tf.float32)

        # Create our optimizer
        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

        iter_count = 1
        best_loss, best_img = float('inf'), None

        loss_weights = (style_weight, content_weight, variation_weight)

        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features,
            'num_style_layers': len(style_layers)
        }
        # For displaying
        num_rows = 2
        num_cols = 5
        display_interval = num_iter/(num_rows*num_cols)
        start_time = time.time()
        global_start = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means
        imgs = []
        for i in range(num_iter):
            # Compute grads
            grads, all_loss = dst.compute_grads(cfg)

            # Get Losses
            loss, style_score, content_score, variation_score = all_loss

            # Step init_image
            opt.apply_gradients([(grads, init_image)])

            # Clip Values between VGG format valuees
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            # Assingn values Clipped to the image (Gradients)
            init_image.assign(clipped)
            end_time = time.time()
            if loss < best_loss:
                # Update best loss and best image from total loss.
                best_loss = loss
                best_img = tools.deprocess_img(init_image.numpy())

            if i % display_interval == 0:
                start_time = time.time()

                # Use the .numpy() method to get the concrete numpy array
                plot_img = init_image.numpy()
                plot_img = tools.deprocess_img(plot_img)
                imgs.append(plot_img)
                IPython.display.clear_output(wait=True)
                IPython.display.display_png(Image.fromarray(plot_img))
                print('Iteration: {}'.format(i))
                print('Total loss: {:.4e}, '
                      'style loss: {:.4e}, '
                      'content loss: {:.4e}, '
                      'time: {:.4f}s'.format(loss, style_score, content_score,
                                             time.time() - start_time))

        print('Total time: {:.4f}s'.format(time.time() - global_start))
        IPython.display.clear_output(wait=True)
        plt.figure(figsize=(14, 4))
        for i, img in enumerate(imgs):
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

        return best_img, best_loss
