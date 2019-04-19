import tensorflow as tf


def gram_matrix(tensor):
    ''''''
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def style_loss(generated_style, gram_target):

    generated_style = generated_style[0]
    gram_target = gram_target[0]

    height, width, channels = generated_style.get_shape().as_list()
    gram_generated_style = gram_matrix(generated_style)
    # / (4. * (channels ** 2) * (width * height) ** 2)
    return tf.reduce_mean(tf.square(gram_generated_style - gram_target))


def content_loss(generated, content):
    ''''''
    return tf.reduce_mean(tf.square(content - generated))


def total_variation_loss(x):
    lote, img_h, img_w, channels = x.get_shape().as_list()
    a = tf.square(x[:, :img_h - 1, :img_w - 1, :] - x[:, 1:, :img_w - 1, :])
    b = tf.square(x[:, :img_h - 1, :img_w - 1, :] - x[:, :img_h - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def calculate_losses(content_package, style_package,
                     weight_style_layer, variation_tensor=None):

    partial_style_loss = 0
    partial_content_loss = 0
    partial_variation_loss = 0

    for index, generate_style, gram_style in enumerate(style_package):
        partial_style_loss += 1. * \
            style_loss(generate_style, gram_matrix)
    for index, generate_content, content_layer in enumerate(content_package):
        partial_content_loss += 1. * \
            content_loss(generate_content, content_layer)
    if variation_tensor is not None:
        partial_variation_loss += total_variation_loss(variation_tensor)
    else:
        partial_variation_loss = 0
    partial_total_loss = total_loss(
        [partial_style_loss, partial_content_loss, partial_variation_loss])

    return (partial_total_loss, partial_style_loss,
            partial_content_loss, partial_variation_loss)


def total_loss(losses):
    ''''''
    t_loss = 0
    for loss in losses:
        t_loss += loss
    return t_loss


def compute_loss(model, loss_weights, init_image, gram_style_features,
                 content_features, num_style_layers):

    style_weight, content_weight = loss_weights

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features,
                                        style_output_features):
        style_score += weight_per_style_layer * \
            style_loss(comb_style, target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(1)
    for target_content, comb_content in zip(content_features,
                                            content_output_features):
        content_score += weight_per_content_layer * \
            content_loss(comb_content, target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score
