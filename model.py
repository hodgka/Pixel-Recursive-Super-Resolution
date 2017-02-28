from __future__ import absolute_import, print_function

import os.path
import time

import numpy as np
import scipy.misc
import tensorflow as tf

import layers
from utils import one_hot

FLAGS = tf.app.flags.FLAGS

<<<<<<< HEAD

class PixelResNet(object):
=======
class PixelResolutionNet(object):

    def __init__(self, input_images, config):
        self.X = input_images
        self.output_shape = [config.batch_size, config.height, config.width, config.channels]
        self.B = config.conditional_layers
        self.prior = prior_network(X, config)
        self.conditional = conditional_network(X, config)
>>>>>>> 1c32f535593de7483faf377efd979be849233740

    def __init__(self, session, features, labels, channels=3):
        self.session = session
        self.features = features
        self.labels = labels
        self.channels = channels

<<<<<<< HEAD
        self.X = input_images
        self.output_shape = [config.batch_size, config.height, config.width, config.channels]
        self.B = config.conditioning_layers
        self.prior = self.prior_network()
        self.conditioning = self.conditioning_network()
        # self.prior = prior_network(X, config)
        # self.conditioning = conditioning_network(X, config)

    def conditioning_network(self):

        # X is a placeholder until tf.session.run() gets called, so use that instead
        # inputs = tf.placeholder(tf.float32, [None, height, width, channels])
        block = self.X

        for _ in range(self.B):
            block = layers.residual_block(block, filter_shape=[3, 3, -1, 32], name="res_block_1")
        block = layers.transposed_conv2d_layer(
            block, filter_shape=[3, 3, -1, 32], output_shape=self.output_shape, name="trans_block_1")
        for _ in range(self.B):
            block = layers.residual_block(block, filter_shape=[3, 3, -1, 32], name="res_block_2")
        block = layers.transposed_conv2d_layer(
            block, filter_shape=[3, 3, -1, 32], output_shape=self.output_shape, name="trans_block_2")
        for _ in range(self.B):
=======
        return tf.sub(one_hot_encoding, ABA_lse)

    def conditional_network(self, config):

        inputs = tf.placeholder(tf.float32, [None, height, width, channels])
        B = 6
        block = inputs
        for _ in range(B):
            block = layers.residual_block(block, filter_shape=[3, 3, -1, 32], name="res_block_1")
        block = layers.transposed_conv2d_layer(
            block, filter_shape=[3, 3, -1, 32], output_shape=self.output_shape, name="trans_block_1")
        for _ in range(B):
            block = layers.residual_block(block, filter_shape=[3, 3, -1, 32], name="res_block_2")
        block = layers.transposed_conv2d_layer(
            block, filter_shape=[3, 3, -1, 32], output_shape=self.output_shape, name="trans_block_2")
        for _ in range(B):
>>>>>>> 1c32f535593de7483faf377efd979be849233740
            block = layers.residual_block(block, filter_shape=[3, 3, -1, 32], name="res_block_3")
        conv = layers.conv_layer(block, filter_shape=[1, 1, -1, 3 * 256])
        return conv

<<<<<<< HEAD
    def prior_network(self, h=None):
        """
        PixelCNN implementation for the prior network
        Inputs should already be preprocessed by this point
        Args:
            h - one hot latent conditioning vector
        Returns:
            prior network to be used as input to a softmax layer
        """

        masked_conv_1 = layers.conv_layer(self.X, [-1, 7, 7, 64], "maked_conv_1", mask='a')

        # rename to make chaining layers easy
        v_stack_in = masked_conv_1
        for i in range(self.prior_layers):
=======
    def prior_network(self, config, h=None):
        """
        PixelCNN implementation for the prior network
        """
        # inputs should already be processed by this point(normalized, whitened, etc..)
        input_shape = self.output_shape
        input_shape[0] = None
        inputs = tf.placeholder(tf.float32, input_shape)

        masked_conv_1 = layers.conv_layer(inputs, [-1, 7, 7, 64], "maked_conv_1", mask='a')

        for i in range(config.prior_layers):
>>>>>>> 1c32f535593de7483faf377efd979be849233740
            filter_shape = [-1, 5, 5, 64]
            # type of convolution mask to use
            mask = "a" if i == 0 else "b"
            i = str(i)
            with tf.variable_scope("v_stack" + i):
                v_stack = layers.gated_cnn_layer(v_stack_in, filter_shape, mask=mask)

        masked_conv_2 = layers.conv_layer(v_stack, [1, 1, -1, 1024], "maked_conv_2", mask='a')
        masked_conv_3 = layers.conv_layer(masked_conv_2, [1, 1, -1, 3 * 256], "maked_conv_3", mask='a')
<<<<<<< HEAD
        return masked_conv_3

    def merge_networks(self):
        """
        merge the results of the prior and conditioning networks with a softmax layers
        Args:
            None
        Returns:
            Super-resolution image of inputs
        """
        prior = self.prior
        conditioning = self.conditioning
        p_yi = tf.nn.softmax(tf.add(prior, conditioning))
        output_image = tf.reshape(p_yi, shape=[-1, 32, 32, 3])
        return output_image

    def train(self, train_data, iterations=200000):
        """
        Train the network on the training data
        Args:
            training_data -
            iterations - number of iterations to train before stopping
        Returns:
            None
        """
        td = train_data

        summaries = tf.merge_all_summaries()
        td.sess.run(tf.initialize_all_variables())

        lr = FLAGS.learning_rate
        start_time = time.time()
        done = False
        batch = 0
        # test_feature, test_label = td.sess.run
        # TODO finish implementing
        test_feature, test_labale = td.sess.run([td.test_features, td.test_labels])

        for i in iterations:
            batch += 1
            feed_dict = {td.learning_rate: lr}

            ops = [self.optimizer, self.loss]
            = td.sess.run(ops, feed_dict=feed_dict)
            if batch % 10 == 0:
                elapsed = (time.time() - start_time) // 60

                print("Progress[{}%%], Batch[{}], Loss[{}]".format(i // iterations, batch, self.loss))

    def loss(self, A_i, B_i):
        """
        Custom loss function described in https://arxiv.org/pdf/1702.00783.pdf
        Args:
            A_i - pixels in conditioning network less than i
            B_i - pixels in prior network less than i
        Returns:
            loss between upscaled image and ground truth
        """
        one_hot_encoding = tf.matmul(one_hot(self.ground_truth), tf.add(2 * A_i, B_i))
        AB_lse = tf.reduce_logsumexp(tf.add(A_i, B_i))
        A_lse = tf.reduce_logsumexp(A_i)
        ABA_lse = tf.add(AB_lse, A_lse)
        self.loss = tf.sub(one_hot_encoding, ABA_lse)
        return self.lossface

    def create_loss(self, real_output, generated_output, features):
        """
        create an instance of the pixelresnet loss function
        Args:

        """
        pass

    def create_optimizer(self, variable_list):
        """
        create a optimizer instance
        """
        self.global_step = tf.variable(0, dtype=tf.int64, trainable=False, name="global_step")
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9,
                                                   epsilon=1e-8).minimize(self.loss, var_list=variable_list)
        return (self.global_step, self.learning_rate, self.optimizer)
=======

        return masked_conv_3

    def combine_networks(self, config):
        prior = self.prior_network
        conditional = self.conditional_network
        p_yi = tf.nn.softmax(tf.add(prior, conditional))
>>>>>>> 1c32f535593de7483faf377efd979be849233740


if __name__ == "__main__":
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    import tensorflow as tf
    x = tf.range(0, 3)
    a, b, c = x
    # x = tf.placeholder(tf.float32, [None, 784])
    # W = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))
    # y = tf.nn.softmax(tf.matmul(x, W) + b)
    # y_ = tf.placeholder(tf.float32, [None, 10])
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
    #                                               tf.log(y), reduction_indices=[1]))
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()
    # for _ in range(1000):
    #     batch_xs, batch_ys = mnist.train.next_batch(100)
    #     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={
    #       x: mnist.test.images, y_: mnist.test.labels}))
