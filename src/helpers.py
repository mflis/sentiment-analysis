# import argparse
#
#
# class ArgParser:
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cut', type=int)
#     parser.add_argument('--epochs', type=int, default=15)
#
#     args = parser.parse_args()
#
#     def maybe_cut_args(self, *arrays):
#         return [(x[:self.args.cut]) for x in arrays]
#
#     def epochs_number(self):
#         return self.args.epochs

from pandas import DataFrame


def flatten(vector):
    return [item for sublist in vector for item in sublist]


def preprocess_scores(vector):
    return [int(score) - 1 for score in vector]


def filter_invalid_data(csv_columns: DataFrame):
    mask = csv_columns.Score.apply(lambda x: x.isnumeric() and int(x) < 6)
    return csv_columns[mask]
