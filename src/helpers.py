import csv
import os

# noinspection PyUnresolvedReferences
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras.python.keras.preprocessing.text import Tokenizer

import src

VOCABULARY_LIMIT = 10000
RANDOM_SEED = 7
TEST_SPLIT = 0.20


def flatten(vector):
    return [item for sublist in vector for item in sublist]


def filter_invalid_data(csv_columns: pd.DataFrame):
    mask = csv_columns.Score.apply(lambda x: x.isnumeric() and int(x) < 6 and int(x) != 3)
    return csv_columns[mask]


def binarize_score(csv_raw: pd.DataFrame):
    return csv_raw.Score.map(lambda x: 1 if int(x) < 3 else 0)


# todo refactor into dict
# df = odf[odf['Score'] != 3]
# X = df['Text']
# y_dict = {1:0, 2:0, 4:1, 5:1}
# y = df['Score'].map(y_dict)

def getColumns(filepath, rows_cut):
    raw_csv = read_csv(filepath, rows_cut)
    cleaned_csv = filter_invalid_data(raw_csv)
    binary_scores = binarize_score(cleaned_csv)
    texts = flatten(cleaned_csv.as_matrix(['Text']).tolist())
    scores = binary_scores.as_matrix().tolist()
    return texts, scores


def dataPath():
    return os.path.join(src.ROOT_DIR, 'data/AmazonReviews-raw.csv')


def read_csv(filepath, rows_cut):
    return pd.read_csv(filepath, encoding='utf8', nrows=rows_cut, quoting=csv.QUOTE_NONE, usecols=['Score', 'Text'],
                       na_filter=False, memory_map=True, dtype={'Score': str, 'Text': str})


def get_tokenizer():
    return Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                     lower=True,
                     split=" ",
                     num_words=VOCABULARY_LIMIT)


def get_test_train_set(row_limit=1000, undersample=False):
    tokenizer = get_tokenizer()
    texts, scores = getColumns(dataPath(), row_limit)

    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    x_train_words, x_test_words, y_train, y_test = train_test_split(texts, scores, test_size=TEST_SPLIT,
                                                                    random_state=RANDOM_SEED)
    if undersample:
        x_train_words, y_train = rus.fit_sample(conv(x_train_words), y_train)
        x_train_words = np.reshape(x_train_words, (-1,))

    tokenizer.fit_on_texts(x_train_words)
    x_train = tokenizer.texts_to_matrix(x_train_words, mode='tfidf')
    x_test = tokenizer.texts_to_matrix(x_test_words, mode='tfidf')
    return (x_train, y_train), (x_test, y_test)


def conv(list_arg):
    return np.asarray(list_arg).reshape(-1, 1)
