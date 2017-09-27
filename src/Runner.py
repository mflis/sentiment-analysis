import numpy as np
import tensorflow as tf

from src import Config, Placeholders, Model, Input
from src.utils import split, get_batches


class Runner(object):
    def __init__(self,
                 config: Config,
                 placeholders: Placeholders,
                 model: Model):
        self.config = config
        self.model = model
        self.placeholders = placeholders
        self.input = Input(self.config)

    def run(self):
        # with graph.as_default():
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('./logs/tb/train', sess.graph)
            test_writer = tf.summary.FileWriter('./logs/tb/test', sess.graph)
            iteration = 1
            self.model.buildGraph()
            inputs, embeddings, labels = self.input.readInput()

            train_x, val_x, test_x = split(inputs)
            train_y, val_y, test_y = split(labels)

            for e in range(self.config.epochs):
                state = sess.run(self.model.initial_state)

                for ii, (x_train_batch, y_train_batch) in \
                        enumerate(get_batches(train_x, train_y, self.config.batch_size), 1):

                    feed = {self.placeholders.inputs: x_train_batch,
                            self.placeholders.labels: y_train_batch[:, None],
                            self.placeholders.keep_prob: 0.5,
                            self.model.initial_state: state}
                    summary, loss, state, _ = sess.run([self.model.merged,
                                                        self.model.cost,
                                                        self.model.final_state,
                                                        self.model.optimizer], feed_dict=feed)

                    train_writer.add_summary(summary, iteration)

                    if iteration % 5 == 0:
                        print("Epoch: {}/{}".format(e, self.config.epochs),
                              "Iteration: {}".format(iteration),
                              "Train loss: {:.3f}".format(loss))

                    if iteration % 25 == 0:
                        val_acc = []
                        val_state = sess.run(self.model.cell.zero_state(self.config.batch_size, tf.float32))
                        for x_val_batch, y_val_batch in get_batches(val_x, val_y, self.config.batch_size):
                            feed = {self.placeholders.inputs: x_val_batch,
                                    self.placeholders.labels: y_val_batch[:, None],
                                    self.placeholders.keep_prob: 1,
                                    self.model.initial_state: val_state}
                            summary, batch_acc, val_state = sess.run([self.model.merged,
                                                                      self.model.accuracy,
                                                                      self.model.final_state], feed_dict=feed)
                            val_acc.append(batch_acc)
                        print("Val acc: {:.3f}".format(np.mean(val_acc)))
                    iteration += 1
                    test_writer.add_summary(summary, iteration)
                    saver.save(sess, "checkpoints/sentiment_manish.ckpt")
            saver.save(sess, "checkpoints/sentiment_manish.ckpt")
