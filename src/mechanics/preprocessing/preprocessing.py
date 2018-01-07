import csv

import pandas as pd


def flatten(vector):
    return [item for sublist in vector for item in sublist]


def filter_invalid_data(csv_columns: pd.DataFrame):
    mask = csv_columns.Score.apply(lambda x: x.isnumeric() and int(x) < 6 and int(x) != 3)
    return csv_columns[mask]


def binarize_score(csv_raw: pd.DataFrame):
    return csv_raw.Score.map(lambda x: 1 if int(x) < 3 else 0)


def getColumns(filepath, rows_cut):
    raw_csv = read_csv(filepath, rows_cut)
    cleaned_csv = filter_invalid_data(raw_csv)
    binary_scores = binarize_score(cleaned_csv)
    texts = flatten(cleaned_csv.as_matrix(['Text']).tolist())
    scores = binary_scores.as_matrix().tolist()
    return texts, scores


def read_csv(filepath, rows_cut):
    return pd.read_csv(filepath, encoding='utf8', nrows=rows_cut, quoting=csv.QUOTE_NONE, usecols=['Score', 'Text'],
                       na_filter=False, memory_map=True, dtype={'Score': str, 'Text': str})


