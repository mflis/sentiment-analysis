import tensorflow as tf

from src import Config, Placeholders


class Model(object):
    def __init__(self, config: Config.Config, placeholders: Placeholders.Placeholders):
        self.config = config
        self.placeholders = placeholders
        self.initial_state: tf.Tensor = None
        self.cost: tf.Tensor = None
        self.optimizer: tf.Tensor = None
        self.merged: tf.Tensor = None
        self.accuracy: tf.Tensor = None

    def buildGraph(self):
        # Create the graph object
        # todo probably not needed
        # tf.reset_default_graph()

        with tf.name_scope("Embeddings"):
            embed = tf.nn.embedding_lookup(self.placeholders.embeddings, self.placeholders.inputs)

        with tf.name_scope("RNN_layers"):
            # Stack up multiple LSTM layers, for deep learning
            cell = tf.contrib.rnn.MultiRNNCell([self.get_single_cell()] * self.config.lstm_layers)
            # Getting an initial state of all zeros
            self.initial_state = cell.zero_state(self.config.batch_size, tf.float32)  # todo maybe sth better than zero?

        with tf.name_scope("RNN_forward"):
            outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=self.initial_state)

        with tf.name_scope('predictions'):
            predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
            tf.summary.histogram('predictions', predictions)

        with tf.name_scope('cost'):
            self.cost = tf.losses.mean_squared_error(self.placeholders.labels, predictions)
            tf.summary.scalar('cost', self.cost)

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cost)

        with tf.name_scope('validation'):
            correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), self.placeholders.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.merged = tf.summary.merge_all()

    def get_single_cell(self):
        # Your basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(self.config.lstm_size, reuse=tf.get_variable_scope().reuse)
        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.placeholders.keep_prob)
        return drop
