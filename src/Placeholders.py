import tensorflow as tf

from src import Config


class Placeholders(object):
    def __init__(self, config: Config, n_words):
        self.config = config

        with self.config.graph.as_default():
            with tf.name_scope('inputs'):
                # possible * num lstml ayers
                self.inputs = tf.placeholder(tf.int32, [None, self.config.PADDING_LENGTH], name="inputs")
                self.labels = tf.placeholder(tf.int32, [None, self.config.N_CLASSES], name="labels")
                self.embeddings = tf.Variable(tf.random_uniform((n_words, self.config.EMBEDDING_SIZE), -1, 1))
                self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
