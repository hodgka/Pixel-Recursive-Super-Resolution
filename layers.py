from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf


def split_and_gate(x):
    """
    Split tensor into two channels of along depth and put the tensors through a PixelCNN gate
    """
    x1, x2 = tf.split(x, 2, -1)
    x1 = tf.nn.tanh(x1)
    x2 = tf.nn.sigmoid(x2)
    return x1 * x2

def gated_cnn_layer(x, state, kernel, name):
    """
    Gated PixelCNN layer for the Prior network
    Args:
        x - Input tensor in nhwc format
        state - state from previous layer
        kernel - height and width of kernel
        name - name for scoping
    Returns:
        gated output, and layer state
    """

    with tf.variable_scope(name):
        in_channel = x.get_shape().as_list()[-1]
        _, kernel_w = kernel

        # left side / state input to layer
        left = conv_layer(state, 2 * in_channel, kernel, mask_type='c', name='left_conv')
        new_state = split_and_gate(left)

        # convolution from left side to right side. state -> output
        left_to_right_conv = conv_layer(left, 2 * in_channel, [1, 1], name="middle_conv")

        # right side / output
        right = conv_layer(x, 2 * in_channel, [1, kernel_w], mask_type='b', name='right_conv1')
        right = right + left_to_right_conv
        new_output = split_and_gate(right)
        new_output = conv_layer(new_output, in_channel, [1, 1], mask_type='b', name='right_conv2')
        new_output = new_output + x

        return new_output, new_state


def conv_layer(x, filters, kernel, strides=1, mask_type=None, name=None):
    '''
    Convolutional layer capable of being masked
    Args:
        x - input tensor in nhwc format
        filters - number of feature maps to use
        kernel - height and width of kernel
        strides - stride size
        mask_type - type of mask to use. Masks using one of the following A/B/C vertical stack mask from https://arxiv.org/pdf/1606.05328.pdf
        name - name for scoping
    Returns:
        2d convolution layer
    '''
    # refactored get_weights and conv_layer into one function so it is much simpler and less fragile
    with tf.variable_scope(name):
        kernel_h, kernel_w = kernel
        in_channel = x.get_shape().as_list()[-1]

        # center coords of kernel/mask
        center_h = kernel_h // 2
        center_w = kernel_w // 2

        if mask_type:
            # using zeros is easier than ones, because horizontal stack
            mask = np.zeros((kernel_h, kernel_w, in_channel, filters), dtype=np.float32)

            # vertical stack only, no horizontal stack
            mask[:center_h, :, :, :] = 1

            if mask_type == 'a':  # no center pixel in mask
                mask[center_h, :center_w, :, :] = 0
            elif mask_type == 'b':  # center pixel in mask
                mask[center_h, :center_w + 1, :, :] = 1
            # else only top part of mask
        else:
            # no mask
            mask = np.ones((kernel_h, kernel_w, in_channel, filters), dtype=np.float32)

        # initialize and mask weights
        weights_shape = [kernel_h, kernel_w, in_channel, filters]

        # can use either initializer
        # weights_initializer = tf.contrib.layers.xavier_initializer()
        weights_initializer = tf.truncated_normal_initializer(stddev=0.1)

        weights = tf.get_variable("weights", shape=weights_shape, dtype=tf.float32, initializer=weights_initializer)
        weights = weights * mask

        bias = tf.get_variable('bias', shape=[filters], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        output = tf.nn.conv2d(x, weights, [1, strides, strides, 1], padding="SAME")
        output = tf.nn.bias_add(output, bias)

    return output


def residual_block(X, out_channels, kernel_shape, strides=1, name=None, training=False):
    '''
    ResNet block from https://arxiv.org/pdf/1512.03385.pdf
    Args:
        X - input tensor in nhwc format
        out_channels - number of output channels to use
        kernel_shape - height and width of kernel
        strides - stride size
        name - name for scoping
    '''
    with tf.variable_scope(name):
        conv1 = conv_layer(X, out_channels, kernel_shape, strides=strides, name="conv1")
        batch_norm1 = tf.layers.batch_normalization(conv1, training=training)
        relu1 = tf.nn.relu(batch_norm1)

        conv2 = conv_layer(relu1, out_channels, kernel_shape, strides=strides, name="conv2")
        batch_norm2 = tf.layers.batch_normalization(conv2, training=training)

        output = X + batch_norm2
        return output
