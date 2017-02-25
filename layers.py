from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_weights(shape, name, mask=None):
    '''
    Helper function to mask weights
    '''
    weights_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=weights_initializer)

    if mask:
        # apply the mask
        filter_mid_x = shape[0] // 2
        filter_mid_y = shape[1] // 2
        mask_filter = np.ones(shape, dtype=np.float32)
        mask_filter[filter_mid_x, filter_mid_y:, :, :] = 0.
        mask_filter[filter_mid_x + 1:, :, :, :] = 0.

        W *= mask_filter
    return W


def get_bias(shape, name):
    return tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer)


def conv_op(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class GatedCNNLayer(object):
    """
    Gated PixelCNN layer for the Prior network
    Methods:
        __init__ - initialize layer properties and call gated_conv
        gated_conv -
        output - return fan_out
    """

    def __init__(self, W_shape, fan_in, gated=True, payload=None, mask=None, activation=True, conditional=True):
        self.fan_in = fan_in
        in_dim = self.fan_in.get_shape()[-1]
        self.W_shape = [W_shape[0], W_shape[1], in_dim, W_shape[2]]
        self.b_shape = W_shape[2]

        self.payload = payload
        self.mask = mask
        self.activation = activation
        self.conditional = conditional

        self.gated_conv()

    def gated_conv(self):
        W_f = get_weights(self.W_shape, "W_vert", mask=self.mask)
        W_g = get_weights(self.W_shape, "W_hor", mask=self.mask)
        if self.conditional is not None:
            h_shape = int(self.conditional.get_shape()[1])
            V_f = get_weights([h_shape, self.W_shape[3]], "V_vert")
            b_f = tf.matmul(self.conditional, V_f)
            V_g = get_weights([h_shape, self.W_shape[3]], "V_hor")
            b_g = tf.matmul(self.conditional, V_g)

            b_f_shape = tf.shape(b_f)
            b_f = tf.reshape(b_f, (b_f_shape[0], 1, 1, b_f_shape[1]))
            b_g_shape = tf.shape(b_g)
            b_g = tf.reshape(b_g, (b_g_shape[0], 1, 1, b_g_shape[1]))
        else:
            b_f = get_bias(self.b_shape, "b_vert")
            b_g = get_bias(self.b_shape, "b_hor")

        conv_f = conv_op(self.fan_in, W_f)
        conv_g = conv_op(self.fan_in, W_g)

        if self.payload is not None:
            conv_f += self.payload
            conv_g += self.payload

        self.fan_out = tf.mul(tf.tanh(conv_f + b_f), tf.sigmoid(conv_g + b_g))

    def output(self):
        return self.fan_out


class ResNetLayer(object):
    pass
