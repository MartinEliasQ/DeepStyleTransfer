# from dts import load_images
import sys
import dst.tools as tools
import numpy as np

import dst.losses as losses
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow import keras
from tensorflow.python.keras import models

import time

K = keras.backend

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

        style_features = [style_layer
                          for style_layer in style_outputs[:num_style_layers]]
        content_features = [content_layer
                            for content_layer
                            in content_outputs[num_style_layers:]]

        return style_features, content_features

    @staticmethod
    def compute_grads(cfg):
        # Create context to follow gradients
        with tf.GradientTape() as tape:
            all_loss = losses._compute_loss(**cfg)
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
                      content_weight=1e5, style_weight=1e-3,
                      num_iter=1000, init_image=None):

        print(content_path, style_path,
              content_layers,
              style_layers,
              content_weight, style_weight,
              num_iter, init_image)

        # Get New Model with the respective outputs(CNN layers)
        model = dst.get_model(content_layers, style_layers)

        # Get  feacture maps for Style and Content
        style_features, content_features = dst._get_feactuere_maps(
            model, content_path, style_path, len(style_layers))
        print(style_features)
        print(content_features)

        # Get Gram Matrix per each Style Layer
        gram_style_features = [losses._gram_matrix(
            style_feature) for style_feature in style_features]

        # Set Init Image
        if init_image is None:
            init_image = tools._load_and_process_img(content_path)
        else:
            init_image = tf.keras.applications.vgg19.preprocess_input(
                init_image)

        # Set in Tensor init image
        init_image = tfe.Variable(init_image, dtype=tf.float32)

        # Create our optimizer
        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

        iter_count = 1
        best_loss, best_img = float('inf'), None

        loss_weights = (style_weight, content_weight)

        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
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

        for i in range(num_iter):
            # Compute grads
            grads, all_loss = compute_grads(cfg)

            # Get Losses
            loss, style_score, content_score = all_loss

            # Step init_image
            opt.apply_gradients([(grads, init_image)])

            # Clip Values between VGG format valuees
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            # Assingn values Clipped to the image (Gradients)
            init_image.assign(clipped)
            end_time = time.time()

            if i % display_interval == 0:
                start_time = time.time()
