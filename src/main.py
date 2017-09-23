#!/usr/bin/env python
# -*- coding: utf-8 -*-
from src import readInput

if __name__ == '__main__':
    data, embeddings, labels = readInput()
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    print('Shape of embedding tensor:', embeddings.shape)
