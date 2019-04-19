import tensorflow as tf


def gram_matrix(tensor):
    ''''''
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def style_loss(generated_style, gram_target):

    height, width, channels = generated_style.get_shape().as_list()
    gram_generated_style = gram_matrix(generated_style)
    # / (4. * (channels ** 2) * (width * height) ** 2)
    return tf.reduce_mean(tf.square(gram_generated_style - gram_target))


def content_loss(generated, content):
    ''''''
    return tf.reduce_mean(tf.square(generated - content))


def total_variation_loss(x):
    lote, img_h, img_w, channels = x.get_shape().as_list()
    a = tf.square(x[:, :img_h - 1, :img_w - 1, :] - x[:, 1:, :img_w - 1, :])
    b = tf.square(x[:, :img_h - 1, :img_w - 1, :] - x[:, :img_h - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def total_loss(losses):
    ''''''
    t_loss = 0
    for loss in losses:
        t_loss += loss
    return t_loss


def compute_loss(model, loss_weights, init_image, gram_style_features,
                 content_features, num_style_layers):

    style_weight, content_weight, variation_weight = loss_weights

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0
    variation_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)

    for target_style, comb_style in zip(gram_style_features,
                                        style_output_features):
        style_score += weight_per_style_layer * \
            style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(1)
    for target_content, comb_content in zip(content_features,
                                            content_output_features):
        content_score += weight_per_content_layer * \
            content_loss(comb_content[0], target_content)

    variation_score = total_variation_loss(init_image)

    style_score *= style_weight
    content_score *= content_weight
    variation_score *= variation_weight
    # Get total loss
    loss = style_score + content_score + variation_score
    return loss, style_score, content_score, variation_score
