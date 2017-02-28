#! /usr/bin/env python
from __future__ import print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import tensorflow as tf

from model import PixelResNet
from utils import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('log_device_placement', False, "Log the device where variables are placed.")
tf.app.flags.DEFINE_float('learning_rate', 0.0004, "Learning Rate")
tf.app.flags.DEFINE_integer('B', 6, "Number of ResNet layers in conditional network")
tf.app.flags.DEFINE_integer('batch_size', 32, "Number of samples per batch")
tf.app.flags.DEFINE_integer('checkpoint_period', 10000, "Number of batches in between checkpoints")
tf.app.flags.DEFINE_integer('condonditional_fmap', 32, "Number of feature maps for the conditional network")
tf.app.flags.DEFINE_integer('image_size', 8, "Size in pixels of image")
tf.app.flags.DEFINE_integer('iterations', 200000, "Number of training iterations")
tf.app.flags.DEFINE_integer('prior_fmap', 64, "Number of feature maps for the prior network")
tf.app.flags.DEFINE_integer('random_seed', 0, "Seed to initialize rng")
tf.app.flags.DEFINE_integer('summary_duration', 500, "Number of batches between summaries")
tf.app.flags.DEFINE_integer('test_vectors', 16, "Number of features to use for testing")
tf.app.flags.DEFINE_integer("gated_cnn_layers", 20, "Number of gated convolutional layers to use in prior network")
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint', "Output folder where checkpoints are dumped.")
tf.app.flags.DEFINE_string('dataset', 'dataset', 'Path to dataset directory')
tf.app.flags.DEFINE_string('train_dir', 'train', "Output folder where training logs are dumped.")


class TrainData(object):
    """
    Helper class to easily collect training data together
    """

    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def setup():
    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    return sess, summary_writer


def create_model(session, features, labels):
    rows = tf.shape(features)[0]
    cols = tf.shape(features)[1]
    channels = tf.shape(features)[2]

    pixel_resnet_input = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, rowls, cols, channels])
    with tf.variable_scope('pixel_resnet') as scope:
        model_temp = PixelResNet(session, features, labels, channels)
        scope.reuse_variables()
        model = PixelResNet(session, model_temp, labels, channels)


def run_model():

    sess, summary_writer = setup()

    # make sure directories are ready for I/O and do a little preprocessing
    all_filenames = prepare_directories(delete_training_dir=True)
    # randomized in prepare_directories, so ready for train/test split
    train_filenames = all_filenames[:-FLAGS.test_vectors]
    test_filenames = all_filenames[-FLAGS.test_vectors:]

    # do remaining preprocessing
    train_features, train_labels = setup_inputs(sess, train_filenames)
    test_features, test_labels = setup_input(sess, test_filenames)

    # set up model
    model = PixelResNet(sess, features, train_labels)
    loss = model.create_loss()
    optimizer = model.create_optimizer()

    # train model
    train_data = TrainData(locals())
    model.train(train_data)


def main(argv=None):
    run_model()

if __name__ == '__main__':
    tf.app.run()
