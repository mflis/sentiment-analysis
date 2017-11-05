# '''Trains an LSTM model on the IMDB sentiment classification task.
# The dataset is actually too small for LSTM to be of any advantage
# compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes:
#
# - RNNs are tricky. Choice of batch size is important,
# choice of loss and optimizer is critical, etc.
# Some configurations won't converge.
#
# - LSTM loss decrease patterns during training can be quite different
# from what you see with CNNs/MLPs/etc.
# '''
# from __future__ import print_function
#
# from tensorflow.contrib.keras.python.keras.datasets import imdb
# from tensorflow.contrib.keras.python.keras.layers import Dense, Embedding
# from tensorflow.contrib.keras.python.keras.layers import LSTM
# from tensorflow.contrib.keras.python.keras.models import Sequential
# from tensorflow.contrib.keras.python.keras.preprocessing import sequence
# from src.helpers import ArgParser
#
# from matplotlib import pyplot
#
# args = ArgParser()
# max_features = 20000
# maxlen = 80  # cut texts after this number of words (among top max_features most common words)
# batch_size = 4096
# epochs = args.epochs_number()
# print('Loading data...')
#
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#
# [x_train, y_train, x_test, y_test] = args.maybe_cut_args(x_train, y_train, x_test, y_test)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
#
# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
#
# print('Build model...')
# model = Sequential()
# model.add(Embedding(max_features, 128))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))
#
# # try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# print('Train...')
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     validation_data=(x_test, y_test))
# score, acc = model.evaluate(x_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)
#
# pyplot.plot(history.history['loss'])
# pyplot.plot(history.history['val_loss'])
# pyplot.title('model train vs validation loss')
# pyplot.ylabel('loss')
# pyplot.xlabel('epoch')
# pyplot.legend(['train', 'validation'], loc='upper right')
# pyplot.savefig("myplot.png", bbox_inches='tight')
