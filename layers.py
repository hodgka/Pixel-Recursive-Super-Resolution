from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf


def gated_cnn_layer(X, W_shape, payload, mask, conditional):
    """
    Gated PixelCNN layer for the Prior network
    From https://github.com/anantzoid/Conditional-PixelCNN-decoder/
    """

    input_dim = X.get_shape()[-1]
    W_shape = [W_shape[0], W_shape[1], input_dim, W_shape[2]]
    b_shape = W_shape[2]

    W_f = get_weights(W_shape, "v_W", mask=mask)
    W_g = get_weights(W_shape, "h_W", mask=mask)

    try:
        # conditional should be a vector of ints
        h_shape = int(conditional.get_shape()[1])
    except TypeError:
        # conditional is None
        b_f = tf.get_variable("v_b", shape=b_shape, dtype=tf.float32, initializer=tf.zeros_initializer)
        b_g = tf.get_variable("h_b", shape=b_shape, dtype=tf.float32, initializer=tf.zeros_initializer)
    else:
        # no error
        V_f = get_weights([h_shape, W_shape[3]], "v_V")
        b_f = tf.matmul(conditional, V_f)
        b_f_shape = tf.shape(b_f)
        b_f = tf.reshape(b_f, (b_f_shape[0], 1, 1, b_f_shape[1]))

        V_g = get_weights([h_shape, W_shape[3]], "h_V")
        b_g = tf.matmul(conditional, V_g)
        b_g_shape = tf.shape(b_g)
        b_g = tf.reshape(b_g, (b_g_shape[0], 1, 1, b_g_shape[1]))

    conv_f = tf.nn.conv2d(X, W_f, strides=[1, 1, 1, 1], padding="SAME")
    conv_g = tf.nn.conv2d(X, W_g, strides=[1, 1, 1, 1], padding="SAME")

    if payload is not None:
        conv_f += payload
        conv_g += payload

    return tf.mul(tf.tanh(conv_f + b_f), tf.sigmoid(conv_g + b_g))


def masked_conv_layer(X, conv_size, strides=[1, 1, 1, 1], padding="SAME"):
    conv_filter = get_weights(conv_size, name, 'a')
    conv_layer = tf.nn.conv2d(X, conv_filter, strides=strides, padding=padding)
    # reshape to 64 channels
    return tf.reshape(conv_layer, [-1, -1, -1, 64])


def get_weights(shape, name, mask=None):
    '''
    Helper function to mask weights
    '''
    weights_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape, tf.float32, weights_initializer)

    if mask:
        filter_mid_x = shape[0] // 2
        filter_mid_y = shape[1] // 2
        mask_filter = np.ones(shape, dtype=np.float32)
        mask_filter[filter_mid_x, filter_mid_y + 1:, :, :] = 0.
        mask_filter[filter_mid_x + 1:, :, :, :] = 0.

        if mask == 'a':  # type A mask vs type B mask
            mask_filter[filter_mid_x, filter_mid_y, :, :] = 0.

        W *= mask_filter
    return W


class GatedCNNLayer(object):
    # TODO cite this better
    # https://github.com/anantzoid/Conditional-PixelCNN-decoder/
    """

    Gated PixelCNN layer for the Prior network
    """

    def __init__(self, W_shape, X, payload=None, mask=None, conditional=None):
        self.X = X
        input_dim = self.X.get_shape()[-1]
        self.W_shape = [W_shape[0], W_shape[1], input_dim, W_shape[2]]
        self.b_shape = W_shape[2]

        self.payload = payload
        self.mask = mask
        self.conditional = conditional

        self.gated_conv()

    def gated_conv(self):
        W_f = get_weights(self.W_shape, "v_W", mask=self.mask)
        W_g = get_weights(self.W_shape, "h_W", mask=self.mask)
        if self.conditional is not None:
            h_shape = int(self.conditional.get_shape()[1])
            V_f = get_weights([h_shape, self.W_shape[3]], "v_V")
            b_f = tf.matmul(self.conditional, V_f)
            V_g = get_weights([h_shape, self.W_shape[3]], "h_V")
            b_g = tf.matmul(self.conditional, V_g)

            b_f_shape = tf.shape(b_f)

            b_f = tf.reshape(b_f, (b_f_shape[0], 1, 1, b_f_shape[1]))
            b_g_shape = tf.shape(b_g)
            b_g = tf.reshape(b_g, (b_g_shape[0], 1, 1, b_g_shape[1]))
        else:
            b_f = get_bias(self.b_shape, "v_b")
            b_g = get_bias(self.b_shape, "h_b")

        conv_f = conv_op(self.X, W_f)
        conv_g = conv_op(self.X, W_g)

        if self.payload is not None:
            conv_f += self.payload
            conv_g += self.payload

        self.fan_out = tf.mul(tf.tanh(conv_f + b_f), tf.sigmoid(conv_g + b_g))

    def output(self):
        return self.fan_out


class ResNetLayer(object):

    def __init__(self, image):

    def res_block(self):

    def transpose_block(self, num_units, mapsize=1, stride=1, stddev=1.0):
        with tf.variable_scope(self.get_layer_str()):
            prev_units = self._get_num_inputs()
