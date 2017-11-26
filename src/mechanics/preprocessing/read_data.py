import os
import re

import numpy as np
from sacred import Ingredient
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from definitions import ROOT_DIR
from .helpers import getColumns

data_ingredient = Ingredient('dataset')


@data_ingredient.config
def my_config():
    sentences_file = 'data/balanced-reviews.csv'
    glove_file = 'data/glove/glove.6B.50d.txt'
    sentences_path = os.path.join(ROOT_DIR, sentences_file)
    glove_path = os.path.join(ROOT_DIR, glove_file)
    max_sequence_length = 100
    dictionary_limit = 40000
    embedding_dim = 50
    rows_cut = 100


@data_ingredient.capture
def load_sentences(sentences_path, rows_cut, dictionary_limit, max_sequence_length):
    texts, labels = getColumns(sentences_path, rows_cut=rows_cut)
    cleaned_texts = [clean_str(x) for x in texts]
    tokenizer = Tokenizer(num_words=dictionary_limit)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=max_sequence_length)
    return labels, word_index, data


@data_ingredient.capture
def prepare_embedding_layer(glove_path, word_index, dictionary_limit, embedding_dim, max_sequence_length):
    embeddings_index = {}
    f = open(glove_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    num_words = min(dictionary_limit, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

            #    note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(input_dim=num_words,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)
    return embedding_layer


def clean_str(string):
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
