#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.Bayes import segmentFileData, count_parameters

if __name__ == '__main__':
    phrases, labels = segmentFileData('data/imbd/train.tsv')
    class_probability, word_occurences, sum_of_words = count_parameters(phrases, labels)
