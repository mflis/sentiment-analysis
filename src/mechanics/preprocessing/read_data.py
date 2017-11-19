import os

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
    sentences_file = 'balanced-reviews.csv'
    glove_file = 'glove/glove.6B.50d.txt'
    BASE_DIR = os.path.join(ROOT_DIR, 'data')
    sentences_path = os.path.join(BASE_DIR, sentences_file)
    glove_path = os.path.join(BASE_DIR, '%s' % glove_file)
    MAX_SEQUENCE_LENGTH = 1000
    dictionary_limit = 20000
    EMBEDDING_DIM = 50
    rows_cut = 100


@data_ingredient.capture
def load_sentences(sentences_path, rows_cut, dictionary_limit, MAX_SEQUENCE_LENGTH):
    texts, labels = getColumns(sentences_path, rows_cut=rows_cut)
    tokenizer = Tokenizer(num_words=dictionary_limit)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return labels, word_index, data


@data_ingredient.capture
def prepare_embedding_layer(glove_path, word_index, dictionary_limit, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
    embeddings_index = {}
    f = open(glove_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    num_words = min(dictionary_limit, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= num_words:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

            #    note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(input_dim=num_words,
                                output_dim=EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer
