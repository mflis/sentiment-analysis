import tensorflow as tf

from src import Config


class Placeholders(object):
    def __init__(self, config: Config):
        self.config = config

        with tf.name_scope('inputs'):
            # possible * num lstml ayers
            self.inputs = tf.placeholder(tf.int32, [None, self.config.PADDING_LENGTH], name="inputs")
            self.labels = tf.placeholder(tf.int32, [None, self.config.N_CLASSES], name="labels")
            self.embeddings = tf.placeholder(tf.int32, [None, self.config.EMBEDDING_SIZE], name="embeddings")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
