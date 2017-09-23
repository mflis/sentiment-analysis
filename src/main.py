#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras

# from tensorflow.contrib.keras.utils import to_categorical
from src import segmentFileData, loadWordVectors

if __name__ == '__main__':
    # sess = tf.InteractiveSession()
    PADDING_LENGTH = 30
    filename_train = 'data/imbd/whole_train.tsv'

    phrases, labels = segmentFileData(filename_train)

    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(phrases)
    sequences = tokenizer.texts_to_sequences(phrases)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    embeddings = loadWordVectors(word_index)
    data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=PADDING_LENGTH)
    data = tf.convert_to_tensor(data)

    labels_array = np.asarray(labels)
    labels = keras.utils.to_categorical(labels_array)
    labels = tf.convert_to_tensor(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    print('Shape of embedding tensor:', embeddings.shape)

    # # split the data into a training set and a validation set
    # indices = np.arange(data.shape[0])
    # np.random.shuffle(indices)
    # data = data[indices]
    # labels = labels[indices]
    # nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    #
    # x_train = data[:-nb_validation_samples]
    # y_train = labels[:-nb_validation_samples]
    # x_val = data[-nb_validation_samples:]
    # y_val = labels[-nb_validation_samples:]
    #
    # # Add print operation
    # a = tf.Print(splitted, [splitted], message="This is a: ")
    #
    # # Add more elements of the graph using a
    # b = a.eval()
