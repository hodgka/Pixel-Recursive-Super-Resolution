import os

import tensorflow as tf

from data import Dataset

FLAGS = tf.app.flags.FLAGS


class ModelTrainer(object):
    """
    Class to take care of setting up model parameters and tensorflow configuration
    """

    def __init__(self, model):
        self.model_dir = FLAGS.model_dir  # directory to write model summaries to
        self.dataset_dir = FLAGS.dataset  # directory containing data
        self.batch_size = FLAGS.batch_size
        self.iterations = FLAGS.iterations
        self.learning_rate = FLAGS.learning_rate

        # create directories if they don"t exist yert
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        self.global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)

        # parse data and create model
        self.dataset = Dataset(self.dataset_dir, self.iterations, self.batch_size)
        self.model = model(self.dataset.hr_images, self.dataset.lr_images)
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 500000, 0.5,  staircase=True)
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8)
        self.train_optimizer = optimizer.minimize(self.net.loss, global_step=self.global_step)

    def train(self):
        init = tf.global_variables_initializer()
        summarize = tf.summary.merge_all()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            # write model summary
            summary_writer = tf.summary.FileWriter(self.model_dir, sess.graph)
            # start input threads to enqueue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord=coord)

            iterations = 1
            try:
                while not coord.should_stop():
                    t1 = time.time()
                    _, loss = sess.run([self.train_optimizer, self.net.loss])
                    t2 = time.time()

                    print("Step {}, loss={} ){} examples/sec; {} sec/batch".format(iterations,
                                                                                   loss, self.batch_size / (t2 - t1), (t2 - t1)))
                    # write summary
                    if iterations % 10 == 0:
                        summary_str = sess.run(summarize)
                        summary_writer.add_summary(summary_str, iterations)

                    # create sample
                    if iterations % 1000 == 0:
                        self.sample_from_model(sess, mu=1.0, step=iterations)

                    # save model
                    if iterations % 10000 == 0:
                        checkpoint_path = os.path.join(self.model_dir, "model.ckpt")
                        saver.save(sess, checkpoint_path, global_step=iterations)

                    iterations += 1

            except tf.errors.OutOfRangeError:
                print("Done training")
            finally:
                coord.request_stop()

            coord.join(threads)

    def sample_from_model(self, session, mu=1.0, step=None):
        """
        Save output image from model
        """
        conditioning_logits = self.net.conditioning_logits
        prior_logits = self.net.prior_logits
        lr_images = self.dataset.lr_images
        hr_images = self.database.hr_images
        np_hr_images, np_lr_images = sess.run([hr_images, lr_images])
        generated_hr_images = np.zeros([self.batch_size, 32, 32, 3], dtype=tf.float32)
        np_conditioning_logits = sess.run(conditioning_logits, feed_dict={lr_images: np_lr_images})

        for i in range(32):
            for j in range(32):
                for c in range(3):
                    np_p_logits = sess.run(prior_logits, feed_dict={hr_images: generated_hr_images})
                    new_pixel = logits_to_pixel(np_conditioning_logits[:, i, j, c * 256: (c + 1) * 256]
                                                + np_prior_logits[:, i, j, c * 256:(c + 1) * 256], mu=mu)
        save_samples(np_lr_images, self.samples_dir + "/lr_" + str(mu * 10) + "_" + str(step) + ".jpg")
        save_samples(np_lr_images, self.samples_dir + "/hr_" + str(mu * 10) + "_" + str(step) + ".jpg")
        save_samples(generated_hr_images, self.samples_dir + "/generate" + str(mu * 10) + "_" + str(step) + ".jpg")
