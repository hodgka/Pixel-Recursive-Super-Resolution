from __future__ import absolute_import, print_function

import os.path
import time

import numpy as np
import scipy.misc
import tensorflow as tf

import layers

FLAGS = tf.app.flags.FLAGS


class PixelResNet(object):
    """
    Pixel Residual Network implementation
    From https://arxiv.org/pdf/1702.00783.pdf
    """

    def __init__(self, lr_images, hr_images, name=None):
        with tf.variable_scope(name):
            self.num_prior_layers = FLAGS.num_prior_layers
            self.B = FLAGS.conditioning_layers  # number of resnet blocks
            self.build_network(lr_images, hr_images)

    def prior_network(self, hr_images):
        """
        Create PixelCNN prior network and return prior logits
        """
        with tf.variable_scope("prior"):
            masked_conv_1 = layers.conv_layer(hr_images, [-1, 7, 7, 64], mask="a", name="conv1")
            v_stack_in = masked_conv_1
            for i in range(self.num_prior_layers):
                v_stack_in = layers.gated_cnn_layer(v_stack_in, [-1, 5, 5, 64], name="gated" + str(i))
            masked_conv_2 = layers.conv_layer(v_stack_in, [1, 1, -1, 1024], mask="a", name="conv2")
            masked_conv_3 = layers.conv_layer(masked_conv_2, [1, 1, 3 * 256], mask="b", name="conv3")

            prior_logits = tf.concat([masked_conv_3[:, :, :, 0::3],
                                      masked_conv_3[:, :, :, 1::3],
                                      masked_conv_3[:, :, :, 2::3]], axis=-1)
            return prior_logits

    def conditioning_network(self, lr_images):
        """
        Create ResNet Conditioning network and return conditioning logits
        """
        with tf.variable_scope("conditioning"):
            block = layers.conv_layer(lr_images, [1, 1, -1, 32], name="conv_init")

            for i in range(2):
                for j in range(self.B):
                    block = layers.residual_block(block, [3, 3, -1, 32], name="res" + str(i) + str(j))
                block = layers.transposed_conv2d_layer(
                    block, [3, 3, -1, 32], output_shape=self.output_shape, name="trans" + str(i))
                block = tf.nn.relu(block)

            for i in range(self.B):
                block = layers.residual_block(block, [3, 3, -1, 32], name="res2" + str(i))

            conditioning_logits = layers.conv_layer(block, [1, 1, -1, 3 * 256], name="conv")

        return conditioning_logits

    def _loss(self, logits, labels):
        """
        Compute cross_entropy loss
        """
        logits = tf.reshape(logits, [-1, 256])
        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, [-1])
        return tf.losses.sparse_softmax_cross_entropy_with_logits(labels, logits)

    def merge_networks(self, lr_images, hr_images):
        """
        Combine Prior and Conditioning networks to generate image and get loss
        """
        labels = hr_images
        hr_images = normalize_color(hr_images)  # convert to [-1, 1] scale
        lr_images = normalize_color(lr_images)  # convert to [-1, 1] scale

        self.prior_logits = self.prior_network(hr_images)
        self.conditioning_logits = self.conditioning_network(lr_images)

        loss1 = self._loss(self.prior_logits + self.conditioning_logits, labels)
        loss2 = self._loss(self.conditioning_logits, labels)
        loss3 = self._loss(self.prior_logits, labels)

        self.loss = loss1 + loss2

        tf.summary.scalar("loss", self.loss)
        tf.summary.scaler("prior_loss", loss3)
