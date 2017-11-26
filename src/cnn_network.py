from __future__ import print_function

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter
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

from mechanics.postprocessing.callbacks import *
from mechanics.postprocessing.scalar_metrics import *
from mechanics.preprocessing.read_data import *

"""
reimplemented in keras from : http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
base paper: https://arxiv.org/abs/1408.5882

"""
cnn_experiment = Experiment('config_cnn', ingredients=[data_ingredient, loggers_ingredient])


# todo code for preprocessing https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py

# todo read-  http://papers.nips.cc/paper/5867-precision-recall-gain-curves-pr-analysis-done-right.pdf
# todo read https://arxiv.org/abs/1510.03820 -  what techniques improve cnn and what does not
# look through CNN content - https://www.udacity.com/course/deep-learning--ud730

# preprocessing as in paper
# disct limit - analyze data and coverage
# glove dimensions 300
#


@cnn_experiment.config
def my_config(dataset, loggers):
    validation_split = 0.2
    max_sequence_length = dataset['max_sequence_length']
    embedding_dim = dataset['embedding_dim']
    keep_prob = 0.5
    nr_of_filters = 10
    filter_sizes = [3, 4, 5]
    batch_size = 128
    epochs = 10


@cnn_experiment.automain
def my_main(max_sequence_length, validation_split, nr_of_filters, embedding_dim, filter_sizes, keep_prob, batch_size,
            epochs):
    cnn_experiment.observers.append(MongoObserver.create())

    sentences_scores, word_index, sentences = load_sentences()
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedding = prepare_embedding_layer(word_index=word_index)(sequence_input)
    expanded_embedding = Lambda(lambda x: K.expand_dims(x, axis=-1))(embedding)

    # todo maybe specify kernel initializers
    convolutions = []
    for filter_size in filter_sizes:
        conv = Conv2D(nr_of_filters, kernel_size=(filter_size, embedding_dim), activation=relu)(expanded_embedding)
        pool = GlobalMaxPooling2D()(conv)
        convolutions.append(pool)
    pooled_layers = concatenate(convolutions)

    dropout = Dropout(keep_prob)(pooled_layers)
    final_layer = Dense(1, activation='sigmoid')(dropout)  # orignal activation was argmax

    model = Model(sequence_input, final_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',  # as in paper, article uses adam
                  metrics=['acc', recall, precision, roc_score])

    with LogFileWriter(cnn_experiment):
        model.fit(sentences, sentences_scores,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=validation_split,
                  callbacks=loggers() + checkpoints())
