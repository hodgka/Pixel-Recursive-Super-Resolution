from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf


# class Dataset(object):
#     """
#     Class to create a image reader queue to batch dataset
#     """

#     def __init__(self, dataset, iterations, batch_size):

#         fnames = [f'./data/{dataset}/{fname}' for fname in os.listdir(f'./data/{dataset}')]

#         filename_queue = tf.train.string_input_producer(fnames)
#         image_reader = tf.WholeFileReader()
#         _, image_file = image_reader.read(filename_queue)

#         image = tf.image.decode_image(image_file, 3)
#         hr_image = tf.image.resize_image_with_crop_or_pad(image, 128, 128)
#         lr_image = tf.image.resize_bilinear(hr_image, (128//8, 128//8))
#         hr_image = tf.image.resize_images(image, [32, 32])  # downsample image
#         lr_image = tf.image.resize_images(image, [8, 8])  # REALLY downsample image
#         hr_image = tf.cast(hr_image, tf.float32)
#         lr_image = tf.cast(lr_image, tf.float32)

#         min_after_dequeue = 1000
#         capacity = min_after_dequeue + 400 * batch_size

#         # batches images of shape [batch_size, 32, 32, 3],[batch_size, 8, 8, 3]
#         self.hr_images, self.lr_images = tf.train.shuffle_batch([hr_image, lr_image], batch_size=batch_size,
#                                                                 min_after_dequeue=min_after_dequeue, capacity=capacity)




class DataQueue:
    def __init__(self, dataset, epochs, batch_size=5, image_size=128, split='train'):
        self.height = self.width = image_size
        self.channels = 3
        self.dataest = dataset
        self.epochs = epochs
        self.split = split

        self.idx = 0
        self.basepath = os.path.join('./data', dataset, split)
        if split=='train':
            self.batch_size = batch_size
        else:
            self.batch_size = 1

        self.base_fnames = sorted([os.path.basename(fname)[:-4] for fname in os.listdir(self.basepath) if 'png' in fname])[:50000]
        self.fnames = [os.path.join(self.basepath, fname + '.png') for fname in self.base_fnames]
        self.batches_per_epoch = len(self.fnames) // self.batch_size
        self.number_of_batches = self.batches_per_epoch * self.epochs
        min_after_dequeue = 100
        capacity = min_after_dequeue + 10 * self.batch_size

        # put filenames into graph by converting to tensors
        fnames = tf.convert_to_tensor(self.fnames, dtype=tf.string)

        # create a queue with a png filename and the corresponding bpg_filename
        queue = tf.train.slice_input_producer([fnames], shuffle=False)

        # grab individual fnames from queue
        ims = tf.read_file(queue[0])

        # read in images
        ims = tf.image.decode_png(ims, 3)

       
        ims = tf.cast(ims, tf.float32)
        ims = (ims / 128.0) - 0.5
        if split == 'train':
            # extract random crops of size image_size from the larger patch
            self.hr_ims = tf.stack([tf.cast(tf.random_crop(ims, [image_size, image_size, 3], seed=i), tf.float32)
                                        for i in range(10)], 0)
            self.lr_ims = tf.image.resize_bilinear(self.hr_ims, (image_size//4, image_size//4))
            self.hr_ims = tf.cast(self.hr_ims, tf.float32)
            self.lr_ims = tf.cast(self.lr_ims, tf.float32)

            min_after_dequeue = 1000
            capacity = min_after_dequeue + 400 * batch_size

            # batches images of shape [batch_size, 32, 32, 3],[batch_size, 8, 8, 3]
            self.hr_images, self.lr_images = tf.train.shuffle_batch([self.hr_ims, self.lr_ims], batch_size=batch_size,
                                                                    min_after_dequeue=min_after_dequeue, capacity=capacity, enqueue_many=True)
        else:
            self.val_ims = tf.train.batch([ims],
                                          batch_size=self.batch_size, 
                                          capacity=capacity, 
                                          enqueue_many=False)