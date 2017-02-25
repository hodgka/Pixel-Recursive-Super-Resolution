"""
Gated PixelCNN and ResNet classes for use in model
"""
from __future__ import absolute_import
from __future__ import print_function

import layers
import numpy as np
import tensorflow as tf


class PixelResolutionNet(object):
    pass


def conditional_network(X, config):
    pass


def prior_network(X, config, h=None):

    height, width, channels = config.height, config.width, config.channels
    # inputs should already be processed by this point(normalized, whitened, etc..)
    inputs = tf.placeholder(tf.float32, [None, height, width, channels])


class PixelCNN(object):
    """
    Prior Network
    """

    def __init__(self, X, conf, h=None):
        self.X = X
        self.X_norm = X
        v_stack_in, h_stack_in = self.X_norm, self.X_norm

        if conf.conditional is True:
            if h is not None:
                self.h = h
            else:
                self.h = tf.placeholder(tf.float32, shape=[None, conf.num_classes])
        else:
            self.h = None

        for i in range(conf.layers):
            filter_size = 3 if i > 0 else 7
            mask = 'b' if i > 0 else 'a'
            residual = True if i > 0 else False
            i = str(i)
            with tf.variable_scope("v_stack" + i):
                v_stack = GatedCNN([filter_size, filter_size, conf.f_map],
                                   v_stack_in, mask=mask, conditional=self.h).output()
                v_stack_in = v_stack

            with tf.variable_scope("v_stack_1" + i):
                v_stack_1 = GatedCNN([1, 1, conf.f_map], v_stack_in, gated=False, mask=mask).output()

            with tf.variable_scope("h_stack" + i):
                h_stack = GatedCNN([1, filter_size, conf.f_map], h_stack_in,
                                   payload=v_stack_1, mask=mask, conditional=self.h).output()

            with tf.variable_scope("h_stack_1" + i):
                h_stack_1 = GatedCNN([1, 1, conf.f_map], h_stack, gated=False, mask=mask).output()
                if residual:
                    h_stack_1 += h_stack_in  # Residual connection
                h_stack_in = h_stack_1

        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, conf.f_map], h_stack_in, gated=False, mask='b').output()

        if conf.data == "mnist":
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, 1], fc1, gated=False, mask='b', activation=False).output()
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.fc2, self.X))
            self.pred = tf.nn.sigmoid(self.fc2)
        else:
            color_dim = 256
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, conf.channel * color_dim], fc1,
                                    gated=False, mask='b', activation=False).output()
                self.fc2 = tf.reshape(self.fc2, (-1, color_dim))

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.fc2, tf.cast(tf.reshape(self.X, [-1]), dtype=tf.int32)))

            '''
                Since this code was not run on CIFAR-10, I'm not sure which
                would be a suitable way to generate 3-channel images. Below are
                the 2 methods which may be used, with the first one (self.pred)
                being more likely.
            '''
            self.pred_sampling = tf.reshape(tf.multinomial(tf.nn.softmax(
                self.fc2), num_samples=1, seed=100), tf.shape(self.X))
            self.pred = tf.reshape(tf.argmax(tf.nn.softmax(self.fc2),
                                             dimension=tf.rank(self.fc2) - 1), tf.shape(self.X))


if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    import tensorflow as tf
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                                  tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={
          x: mnist.test.images, y_: mnist.test.labels}))
