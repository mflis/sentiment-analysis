#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

if __name__ == '__main__':
    lstm_size = 256
    lstm_layers = 2
    batch_size = 1000
    learning_rate = 0.01

    n_words = 1000 + 1  # Add 1 for 0 added to vocab

    # Create the graph object
    tf.reset_default_graph()
    with tf.name_scope('inputs'):
        inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
        labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Size of the embedding vectors (number of units in the embedding layer)
    embed_size = 300

    with tf.name_scope("Embeddings"):
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)


    def lstm_cell():
        # Your basic LSTM cell
        print(tf.get_variable_scope().reuse)
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
        # Add dropout to the cell
        return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)


    with tf.name_scope("RNN_layers"):
        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.name_scope("RNN_forward"):
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
        tf.summary.histogram('predictions', predictions)
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels_, predictions)
        tf.summary.scalar('cost', cost)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    merged = tf.summary.merge_all()

    with tf.name_scope('validation'):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred + 1, tf.float32))
