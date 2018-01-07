from __future__ import print_function

from sacred import Experiment
from tensorflow.python.keras import Input
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.convolutional import Conv2D

from mechanics.postprocessing.loggers import *
from mechanics.preprocessing.read_data import *

"""
CNN netwrok for sentiment classification based on paper: https://arxiv.org/abs/1408.5882

reimplemented in keras from:
 http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
"""
cnn_experiment = Experiment('config_cnn', ingredients=[data_ingredient, loggers_ingredient])


@cnn_experiment.config
def cnn_config(dataset, loggers):
    max_sequence_length = dataset['max_sequence_length']
    embedding_dim = dataset['embedding_dim']
    embedding_trainable = dataset["embedding_trainable"]
    keep_prob = 0.5
    nr_of_filters = 10
    filter_sizes = [3, 4, 5]
    batch_size = 64
    epochs = 10
    tag = "batch={};f_sizes={};f_num={};embed_dim={};" \
          "epochs={};embed_trainable={}".format(batch_size, filter_sizes,
                                                nr_of_filters,
                                                embedding_dim, epochs,
                                                embedding_trainable)


@cnn_experiment.automain
def cnn_main(max_sequence_length, nr_of_filters, embedding_dim,
             filter_sizes, keep_prob, batch_size, epochs, tag):
    sentences_scores, word_index, sentences = load_sentences()

    sequence_input = Input(name="input_sentences",
                           shape=(max_sequence_length,), dtype='int32')
    embedding = prepare_embedding_layer(word_index=word_index)(sequence_input)
    expanded_embedding = Lambda(name="add_channel_dim", function=lambda x: K.expand_dims(x, axis=-1))(embedding)

    convolutions = []
    for index, filter_size in enumerate(filter_sizes):
        conv = Conv2D(nr_of_filters, name="filter_width_{}_index{}".format(filter_size, index),
                      kernel_size=(filter_size, embedding_dim), activation=relu)(expanded_embedding)
        pool = GlobalMaxPooling2D()(conv)
        convolutions.append(pool)
    pooled_layers = concatenate(convolutions)

    dropout = Dropout(keep_prob)(pooled_layers)
    final_layer = Dense(1, activation='sigmoid')(dropout)

    model = Model(sequence_input, final_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['acc'])

    model.fit(sentences, sentences_scores,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=loggers(tag=tag))
