#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import os
import time

import numpy as np
import tensorflow as tf

from model import model, loss_
from data import *
from utils import sample_from_model

np.random.seed(0)
tf.set_random_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=4e-4, type=float, help='learning_rate')
parser.add_argument('-B', default=6, type=int, help='Number of resnet layers in conditional network')
parser.add_argument('--batch_size', default=8, type=int, help='Number of images per minibatch')
parser.add_argument('--image_size', default=32, type=int, help='Size of high resolution image')
parser.add_argument('--epochs', default=2000, type=float, help='Number of epochs to train for')
parser.add_argument('--dataset', default='CelebA', type=str, help='Dataset to use for training. [CLIC | CelebA]')
parser.add_argument('--model_dir', default='models', type=str, help='Directory to save models to')
# parser.add_argument('--sample_dir', default='samples', type=str, help='Directory to save samples to.')
args = parser.parse_args()

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
print(args.batch_size)
global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
data = DataQueue(args.dataset, args.epochs, args.batch_size, args.image_size)

model = tf.make_template('model', model)
prior_logits, conditioning_logits = model(data.lr_images, data.hr_images)
print(data.hr_images)
loss1 = loss_(prior_logits + conditioning_logits, data.hr_images)
loss2 = loss_(conditioning_logits, data.hr_images)
loss3 = loss_(prior_logits, data.hr_images)
loss = loss1 + loss2

with tf.name_scope("losses"):
    tf.summary.scalar("0_total_loss", loss)
    tf.summary.scalar('1_combined_loss', loss1)
    tf.summary.scalar('2_cond_loss', loss2)
    tf.summary.scalar("3_prior_loss", loss3)


lr_placeholder = tf.placeholder(tf.float32, [], name="lr")
learning_rate = tf.train.exponential_decay(lr_placeholder, global_step, 500000, 0.5, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

summary_op = tf.summary.merge_all()

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
run_config.allow_soft_placement = True
with tf.Session(config=run_config) as sess:
    start = time.time()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(args.model_dir, sess.graph)
    
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    iterations = 1
    try:
        while not coord.should_stop():
            t1 = time.time()
            _, loss_ = sess.run([opt, loss], {lr_placeholder: args.lr})
            t2 = time.time()

            print("Step {}, loss={:.2f}, ({:.1f} examples/sec; {:.3f} sec/batch)".format(iterations,
                                                                                            loss_, args.batch_size / (t2 - t1), (t2 - t1)))
            # summarize model
            if iterations % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, iterations)

            # create sample
            if iterations % 200 == 0:
                print("Sampling from model")
                sample_from_model(sess, conditioning_logits, prior_logits, data, args, step=iterations)
                summary_writer.add_summary
                print("Done sampling model")

            # save model
            if iterations % 10000 == 0:
                checkpoint_path = os.path.join(args.model_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=iterations)

            iterations += 1

    except tf.errors.OutOfRangeError:
        print("Done training")
    finally:
        coord.request_stop()
        coord.join(threads)
