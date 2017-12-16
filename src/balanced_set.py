import os

from definitions import ROOT_DIR
from mechanics.preprocessing.helpers import read_csv


def filter_scores(scores, csv):
    return csv.Score.apply(lambda x: x.isnumeric() and int(x) in scores)


def split(csv, split_point):
    return csv.head(int(len(csv) * (1 - split_point))), csv.tail(int(len(csv) * split_point))


def save(csv, name):
    csv.sample(frac=1).to_csv(os.path.join(data_dir, name), index=False, index_label=["Score", "Text"])


data_dir = os.path.join(ROOT_DIR, 'data')
filepath = os.path.join(data_dir, 'AmazonReviews.csv')
raw_csv = read_csv(filepath, 1000000)

negatives = raw_csv[filter_scores([1, 2], raw_csv)]
positives_all = raw_csv[filter_scores([4, 5], raw_csv)]
positives = positives_all.head(len(negatives))

pos_train, pos_val = split(positives, 0.2)
neg_train, neg_val = split(negatives, 0.2)

save(pos_train.append(neg_train), 'balanced-reviews_train.csv')
save(pos_val.append(neg_val), 'balanced-reviews_val.csv')
