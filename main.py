#! /usr/bin/env python
from __future__ import print_function

import argparse
import glob
import logging
import os

import numpy as np
import tensorflow as tf

from model import PixelResolutionNet as Net
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help="Path to dataset. Defaults to current directory")
    parser.add_argument('--cond_f_map', type=int, default=32, help="Number of feature maps for the conditional network")
    parser.add_argument('--prior_f_map', type=int, default=64, help="Number of feature maps for the prior network")
    parser.add_argument('--prior_layers', type=int, default=20, help="Number of layers to use in prior network")
    parser.add_argument('--iters', type=int, default=200000, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.0004, help="Learning Rate")
    parser.add_argument('--batch_size', type=int, default=32, help="Size of training batches")
    parser.add_argument('--grad_clip', type=int, default=1, help="Gradient clipping")
    parser.add_argument('--stride', type=int, default=1, help="Stride to use for convolutions")
    # parser.add_argument('--opt', type=str, default="RMS", help="Optimizer to use for learning")
    parser.add_argument('--summary_path', type=str, default='logs', help="Directory path to store log files")
    return parser.parse_args()


def train(data, config):
    X = tf.placeholder(tf.float32, shape=[None, config.input_height, config.input_width, config.channels])
    model = PixelResolutionNet(X, config)
    if config.
    trainer = tf.train.RMSPropOptimizer(decay=0.95, momentum=0.9, epsilon=1e-8)
    gradients = Optimizer.compute_gradients(Net.loss)

    clipped_gradients = [(tf.clip_by_value(_[0], -config.grad_clip, config.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session as sess:
        sess.run(tf.initialize_all_variables())
        if os.path.exists(config.model_path):
            saver.restore(sess, config.model_file)
            print("Reusing model")

        print("Starting training...")

        counter = 0
        for i in range(config.epochs):
            for j in range(config.num_batches):
                batch_X, counter = get_batch(data, counter, config.batch_size)
                data_dict = {X: batch_X}
                data_dict[model.h] = batch_y
                _, cost = sess.run([optimizer, Net.loss], feed_dict=data_dict)
            print("Epoch: {}, Cost: {}".format(i, cost))

if __name__ == '__main__':
    args = parse_arguments()
    data = load_images(args.data)
