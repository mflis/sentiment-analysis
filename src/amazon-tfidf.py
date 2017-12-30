from sacred import Experiment
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.regularizers import l2

from mechanics.postprocessing.callbacks import loggers_ingredient, loggers
from mechanics.preprocessing.read_data import data_ingredient, get_data_tfidf

tfidf_experiment = Experiment('config_tfidf', ingredients=[data_ingredient, loggers_ingredient])


@tfidf_experiment.config
def my_config(dataset, loggers):
    batch_size = 64
    epochs = 10
    vocabulary_limit = dataset['vocabulary_limit']
    tag = "batch={};epochs={};".format(batch_size, epochs)


@tfidf_experiment.automain
def tfidf_main(vocabulary_limit, epochs, batch_size, tag):
    (x_train, y_train), (x_test, y_test) = get_data_tfidf()

    model = Sequential()

    model.add(Dense(vocabulary_limit, input_dim=vocabulary_limit, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), callbacks=loggers(tag=tag))

# todo  read about statistical significance
# todo to read: http://cs231n.github.io/neural-networks-3/
# activations
# intermediate: relu, elu, selu, tanh, sigmoid
# last: softmax, sigmoid


# todo : to try use top 10 k words - https://www.kaggle.com/ruzerichards/predicting-amazon-reviews-using-cnns
