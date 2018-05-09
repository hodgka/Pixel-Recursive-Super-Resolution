from __future__ import absolute_import, print_function

import os.path
import time

import numpy as np
import scipy.misc
import tensorflow as tf

from layers import *

FLAGS = tf.app.flags.FLAGS


def model(lr_images, hr_images):

    with tf.variable_scope("prior_network"):
        pnet = hr_images
        pnet = conv_layer(pnet, 64, [7, 7], mask_type='a', name='conv1')
        state = pnet
        for i in range(20):
            pnet, state = gated_cnn_layer(pnet, state, [5, 5], name="gated" + str(i))
        pnet = conv_layer(pnet, 1024, [1, 1], mask_type='b', name='conv2')
        pnet = tf.nn.relu(pnet)
        pnet = conv_layer(pnet, 3 * 256, [1, 1], mask_type='b', name="conv3")
        prior_logits = tf.concat([pnet[:, :, :, 0::3],
                                  pnet[:, :, :, 1::3],
                                  pnet[:, :, :, 2::3]], axis=-1)
    
    with tf.variable_scope('conditioning'):
        B = 3
        cnet = lr_images
        cnet = conv_layer(cnet, 32, [1, 1], mask_type=None, name="conv1")
        print('first', cnet)
        for i in range(2):
            for j in range(B):
                cnet = residual_block(cnet, 32, [3, 3], name='residual_{}_{}'.format(str(i + 1), str(j + 1)))
                print('res_{}_{}'.format(i, j), cnet)
            cnet = tf.layers.conv2d_transpose(cnet, 32, [3, 3], 2, padding='same', name="deconv_" + str(i))
            cnet = tf.nn.relu(cnet)
            print('deconv_{}'.format(i), cnet)

        for i in range(B):
            cnet = residual_block(cnet, 32, [3, 3], name='residual_3_' + str(i))
            print('res_{}'.format(i), cnet)
        conditioning_logits = conv_layer(cnet, 3 * 256, [1, 1], mask_type=None, name="conv2")
        print('cond_{}'.format(i), cnet)
    return prior_logits, conditioning_logits

def loss_(logits, labels):
    logits = tf.reshape(logits, [-1, 256])
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [-1])
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return loss


# class PixelResNet(object):
#     """
#     Pixel Residual Network implementation
#     From https://arxiv.org/pdf/1702.00783.pdf
#     """

#     def __init__(self, hr_images, lr_images, name='pixelresnet'):
#         with tf.variable_scope(name):
#             self.B = FLAGS.B  # number of resnet blocks
#             self.merge_networks(hr_images, lr_images)

#     def merge_networks(self, hr_images, lr_images):
#         """
#         Combine Prior and Conditioning networks to generate image and get loss
#         Args:
#             hr_images - high resolution images, ground truth 32x32 images
#             lr_images - low resolution images, down sampled 8x8 images
#         Returns:
#             None, computes loss
#         """
#         labels = hr_images  # high resolution images are ground truth for loss function
#         hr_images = normalize_color(hr_images)  # convert to [-1, 1] scale
#         lr_images = normalize_color(lr_images)  # convert to [-1, 1] scale

#         self.prior_logits = self.prior_network(hr_images)
#         self.conditioning_logits = self.conditioning_network(lr_images)

#         loss1 = self._loss(self.prior_logits + self.conditioning_logits, labels)
#         loss2 = self._loss(self.conditioning_logits, labels)
#         loss3 = self._loss(self.prior_logits, labels)

#         self.loss = loss1 + loss2

#         tf.summary.scalar("loss", self.loss)
#         tf.summary.scalar("prior_loss", loss3)

#     def prior_network(self, hr_images):
#         """
#         Create PixelCNN prior network and return prior logits
#         Args:
#             hr_images - high resolution images - 32x32
#         Returns:
#             logits for prior network
#         """
#         with tf.variable_scope("prior"):
#             conv_1 = conv_layer(hr_images, 64, 7, mask_type='a', name='conv1')
#             X = state = conv_1
#             for i in range(20):
#                 X, state = gated_cnn_layer(X, state, 5, name="gated" + str(i))
#             conv2 = conv_layer(X, 1024, [1, 1], mask_type='b', name='conv2')
#             conv2 = tf.nn.relu(conv2)
#             prior_logits = conv_layer(conv2, 3 * 256, 1, mask_type='b', name="conv3")
#             prior_logits = tf.concat([prior_logits[:, :, :, 0::3],
#                                       prior_logits[:, :, :, 1::3],
#                                       prior_logits[:, :, :, 2::3]], axis=-1)

#             return prior_logits

#     def conditioning_network(self, lr_images):
#         """
#         Create ResNet Conditioning network

#         """
#         with tf.variable_scope('conditioning') as scope:
#             inputs = lr_images
#             inputs = conv_layer(inputs, 32, 1, mask_type=None, name="conv_init")
#             for i in range(2):  # easy to make it deeper this way
#                 for j in range(self.B):
#                     inputs = residual_block(inputs, 32, 3, name='residual_{}_{}'.format(str(i + 1), str(j + 1)))
#                 inputs = tf.layers.conv2d_transpose(inputs, 32, 3, 2, name="deconv_" + str(i))
#                 inputs = tf.nn.relu(inputs)

#             for i in range(self.B):
#                 inputs = residual_block(inputs, 32, 3, name='residual_3_' + str(i))

#             conditioning_logits = conv_layer(inputs, 3 * 256, [1, 1], mask_type=None, name="conv")
#             return conditioning_logits

#     def _loss(self, logits, labels):
#         """
#         Compute cross_entropy loss
#         Args:
#             Logits - sum from conditioning and prior networks
#             labels - ground truth from 32 x 32 images
#         Returns:
#             cross_entropy loss over image
#         """
#         logits = tf.reshape(logits, [-1, 256])
#         labels = tf.cast(labels, tf.int32)
#         labels = tf.reshape(labels, [-1])
#         loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
#         return loss
