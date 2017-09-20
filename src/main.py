#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.Bayes import segmentFileData, count_parameters, doc_tem_matrix

if __name__ == '__main__':
    train_phrases, train_labels = segmentFileData('data/imbd/train.tsv')
    class_probability, word_occurences, sum_of_words = count_parameters(train_phrases, train_labels)

    test_phrases, test_labels = segmentFileData('data/imbd/test.tsv')
    matrix = doc_tem_matrix(test_phrases)
