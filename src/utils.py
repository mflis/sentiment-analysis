import csv

import numpy as np


def segmentFileData(filename):
    file = open(filename, newline='')
    spam_reader = csv.DictReader(file, delimiter='\t')
    labels = []
    phrases = []
    for row in spam_reader:
        labels.append(int(row['Sentiment']))
        phrases.append(row['Phrase'])
    return phrases, labels


DEFAULT_FILE_PATH = "data/glove/glove.6B.50d.txt"


def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """Read pretrained GloVe vectors"""
    wordVectors = np.zeros((len(tokens) + 1, dimensions))
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[tokens[token]] = np.asarray(data)
    return wordVectors
