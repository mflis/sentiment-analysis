import tensorflow as tf


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    PADDING_LENGTH = 30
    EMBEDDING_SIZE = 50
    DEFAULT_FILE_PATH = "data/glove/glove.6B.50d.txt"
    TRAIN_FILE = 'data/imbd/whole_train.tsv'
    N_CLASSES = 5
    # TODO shouln't this match PADDING_LENGTH ?
    lstm_size = 256
    lstm_layers = 2
    batch_size = 500
    learning_rate = 0.001
    epochs = 10
    graph = tf.Graph()
