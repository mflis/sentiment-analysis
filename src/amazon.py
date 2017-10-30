import csv
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.python.keras.utils import np_utils

import src
from src.helpers import *

dataPath = os.path.join(src.ROOT_DIR, 'data/AmazonReviews-raw.csv')
nr_of_classes = 5
seed = 7

rawCSV = pd.read_csv(dataPath, encoding='utf8', nrows=20000, quoting=csv.QUOTE_NONE, usecols=['Score', 'Text'],
                     na_filter=False, memory_map=True, dtype={'Score': str, 'Text': str})
cleanedCSV = filter_invalid_data(rawCSV)
texts = flatten(cleanedCSV.as_matrix(['Text']).tolist())
scores = flatten(cleanedCSV.as_matrix(['Score']).tolist())

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      num_words=5000)

score_vector = np_utils.to_categorical(preprocess_scores(scores), nr_of_classes)
X_train_words, X_test_words, y_train, y_test = train_test_split(texts, score_vector, test_size=0.20, random_state=7)

tokenizer.fit_on_texts(X_train_words)

X_train = tokenizer.texts_to_matrix(X_train_words, mode='tfidf')
X_test = tokenizer.texts_to_matrix(X_test_words, mode='tfidf')

model = Sequential()
model.add(Dense(5, activation='relu', input_dim=5000))
model.add(Dense(nr_of_classes, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=4096, epochs=10, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])


# todo: turn  score into polarity

# activations
# intermediate: relu, elu, selu, tanh, sigmoid
# last: softmax, sigmoid
