from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf


def gated_cnn_layer(X, filter_shape, payload=None, mask=None, conditional=None, name=None):
    """
    Gated PixelCNN layer for the Prior network
    Args:
        X - Input
        filter_shape - Shape of weight tensor in format - [mapsize, mapsize, input_channels, output_channels]
        payload - Vertical stack output
        mask - Type of mask to use (type A / type B)
        conditional - Latent one-hot representation vector, h
    """
    with tf.variable_scope(name):
        b_shape = filter_shape[-1]  # TODO need to figure out whether to use 3rd or 4th entry
        # set filter_shape input channels to number of output channels of X
        # filter_shape[2] = tf.shape(X)[-1]

        W_f = get_weights("v_W", filter_shape, mask=mask)
        W_g = get_weights("h_W", filter_shape, mask=mask)

        if conditional:
            h_shape = tf.shape(conditional)[1]

            V_f = get_weights("v_V", [h_shape, filter_shape[3]])
            b_f = tf.matmul(conditional, V_f)
            b_f_shape = tf.shape(b_f)
            b_f = tf.reshape(b_f, (b_f_shape[0], 1, 1, b_f_shape[1]))

            V_g = get_weights("h_V", [h_shape, filter_shape[3]])
            b_g = tf.matmul(conditional, V_g)
            b_g_shape = tf.shape(b_g)
            b_g = tf.reshape(b_g, (b_g_shape[0], 1, 1, b_g_shape[1]))
        else:
            b_f = tf.get_variable("v_b", shape=b_shape, dtype=tf.float32)
            b_g = tf.get_variable("h_b", shape=b_shape, dtype=tf.float32)

        conv_f = tf.nn.conv2d(X, W_f, strides=[1, 1, 1, 1], padding="SAME")
        conv_g = tf.nn.conv2d(X, W_g, strides=[1, 1, 1, 1], padding="SAME")

        if payload is not None:
            conv_f += payload
            conv_g += payload

    return tf.multiply(tf.tanh(conv_f + b_f), tf.sigmoid(conv_g + b_g))


def conv_layer(X, filter_shape, strides=(1, 1, 1, 1), padding="SAME", mask=None, name=None):
    """
    Convolutional layer that supports A/B PixelCNN masking
    Args:
        name - name scope
        X - output of previous layer
        filter_shape - shape of filter in format [mapsize, mapsize, input_channels, output_channels]
        strides - stride of convolution
        padding - padding of convolution
        mask - Type of mask to apply to filter weights (A or B). Only applied if truthy
    """
    with tf.variable_scope(name):
        conv_filter = get_weights("conv_weights", filter_shape, mask=mask)
        conv = tf.nn.conv2d(X, conv_filter, strides=strides, padding=padding)

        b = tf.get_variable("bias", shape=filter_shape[-1:], dtype=tf.float32, initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)
    return conv


def get_weights(name, filter_shape, mask=None):
    """
    Helper function to mask weights
    Args:
        name - namespace scope
        filter_shape - shape of filter in format [mapsize, mapsize, input_channels, output_channels]
        mask - type of mask to apply to the convolution weights. Mask is only applied if truthy
    """
    weights_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape=filter_shape, dtype=tf.float32, initializer=weights_initializer)

    if mask:
        filter_mid_x = filter_shape[0] // 2
        filter_mid_y = filter_shape[1] // 2
        mask_filter = np.ones(filter_shape, dtype=np.float32)
        mask_filter[filter_mid_x, filter_mid_y + 1:, :, :] = 0.
        mask_filter[filter_mid_x + 1:, :, :, :] = 0.

        if mask == "a":  # type A mask vs type B mask
            mask_filter[filter_mid_x, filter_mid_y, :, :] = 0.

        W *= mask_filter
    return W


def residual_block(X, filter_shape, num_layers=2, name=None):
    """
    ResNet block for the conditioning network
    """
    with tf.variable_scope(name):
        bypass = X  # residual link
        input_channels = filter_shape[-2]
        output_channels = filter_shape[-1]
        # mismatched dimensions -> must preform projection mapping
        if input_channels != output_channels:
            conv = conv_layer(X, filter_shape=[1, 1, input_channels, output_channels], name='projection_conv')
        else:
            conv = X

        for i in range(num_layers):
            batch = tf.contrib.layers.batch_norm(conv, scale=False)  # next layer is ReLU so no need for gamma
            relu = tf.nn.relu(batch)
            conv = conv_layer(relu, filter_shape=filter_shape, name="relu_conv" + str(i))

    return tf.add(conv, bypass)


def transposed_conv2d_layer(X, filter_shape, output_shape, strides=(1, 2, 2, 1), padding="SAME", name=None):
    """
    transpose convolution to upscale input image
    """
    with tf.variable_scope(name):
        # conv_filter has shape [f_height, f_width, in_channels, out_channels]
        conv_filter = get_weights("transposed_conv2d_layer", filter_shape, mask=None)
        # transpose to [f_height, f_width, out_channels, in_channels]
        conv_filter = tf.transpose(conv_filter, perm=[0, 1, 3, 2])

        # make sure that output_shape is scaled correctly
        output_shape = [a * b for a, b in zip(output_shape, strides)]
        out = tf.nn.conv2d_transpose(X, conv_filter, output_shape=output_shape, strides=strides, padding=padding)
        bias = tf.get_variable("bias", output_shape[-1:])
        out = tf.nn.bias_add(out, bias)
    return out
