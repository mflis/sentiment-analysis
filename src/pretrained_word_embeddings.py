from __future__ import print_function

from sacred import Experiment
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential

from mechanics.postprocessing.loggers_prep import *
from mechanics.postprocessing.scalar_metrics import *
from mechanics.preprocessing.read_data import *

cnn_experiment = Experiment('config_demo', ingredients=[data_ingredient, loggers_ingredient])


@data_ingredient.config
def conf():
    MAX_SEQUENCE_LENGTH = 30


@cnn_experiment.config
def my_config(dataset, loggers):
    VALIDATION_SPLIT = 0.2
    MAX_SEQUENCE_LENGTH = dataset['MAX_SEQUENCE_LENGTH']


# todo reimplement this - http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

@cnn_experiment.automain
def my_main(MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT):
    sentences_scores, word_index, sentences = load_sentences()
    embedding = prepare_embedding_layer(word_index=word_index)

    print('input')
    print(sentences_scores)
    print(sentences[0])

    model = Sequential()
    model.add(embedding)
    model.add(Flatten())
    # model.add(Conv1D(128, 5, activation='relu'))
    # model.add(GlobalMaxPooling1D())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', recall, precision, roc_score])

    model.fit(sentences, sentences_scores,
              batch_size=128,
              epochs=10,
              validation_split=VALIDATION_SPLIT,
              callbacks=prepare_loggers())


    # todo read-  http://papers.nips.cc/paper/5867-precision-recall-gain-curves-pr-analysis-done-right.pdf
    # przejrzec pod kątem CNN - https://www.udacity.com/course/deep-learning--ud730
