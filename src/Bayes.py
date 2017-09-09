import csv
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def count_parameters(phrases, labels):
    vectorizer = CountVectorizer()
    word_occurences = vectorizer.fit_transform(phrases)
    class_probability = dict()
    class_word_occurences = dict()
    class_sum_of_words = dict()
    for x in range(0, 5):
        class_probability[x] = Counter(labels)[x] / len(labels)  # P(C)
        class_indexes = np.where(np.array(labels) == x)
        class_word_occurences[x] = np.sum(word_occurences[class_indexes], axis=0)  # count(C,d)
        class_sum_of_words[x] = np.sum(class_word_occurences[x])  # V_c
    return class_probability, class_word_occurences, class_sum_of_words


def segmentFileData(filename):
    file = open(filename, newline='')
    spam_reader = csv.DictReader(file, delimiter='\t')
    labels = []
    phrases = []
    for row in spam_reader:
        labels.append(int(row['Sentiment']))
        phrases.append(row['Phrase'])
    return phrases, labels
