import csv

import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras


def segmentFileData(filename):
    file = open(filename, newline='')
    spam_reader = csv.DictReader(file, delimiter='\t')
    labels = []
    phrases = []
    for row in spam_reader:
        labels.append(int(row['Sentiment']))
        phrases.append(row['Phrase'])
    return phrases, labels


DEFAULT_FILE_PATH = "data/glove/glove.6B.50d.txt"


def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """Read pretrained GloVe vectors"""
    wordVectors = np.zeros((len(tokens) + 1, dimensions))
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[tokens[token]] = np.asarray(data)
    return wordVectors


def readInput():
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
    return data, embeddings, labels
