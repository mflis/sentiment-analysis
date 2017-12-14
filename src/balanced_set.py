import os

from definitions import ROOT_DIR
from mechanics.preprocessing.helpers import read_csv

data_dir = os.path.join(ROOT_DIR, 'data')
filepath = os.path.join(data_dir, 'AmazonReviews.csv')
raw_csv = read_csv(filepath, 1000000)
negatives_mask = raw_csv.Score.apply(lambda x: x.isnumeric() and int(x) in [1, 2])
positives_mask = raw_csv.Score.apply(lambda x: x.isnumeric() and int(x) in [4, 5])
negatives = raw_csv[negatives_mask]
positives = raw_csv[positives_mask]
positives_cut = positives.head(len(negatives))
pos_val = positives_cut.head(int(len(positives_cut) * 0.2))
pos_train = positives_cut.tail(int(len(positives_cut) * 0.8))
neg_val = negatives.head(int(len(negatives) * 0.2))
neg_train = negatives.tail(int(len(negatives) * 0.8))
pos_train.append(neg_train).sample(frac=1).to_csv(os.path.join(data_dir, 'balanced-reviews_train.csv'), index=False,
                                                  index_label=["Score", "Text"])
pos_val.append(neg_val).sample(frac=1).to_csv(os.path.join(data_dir, 'balanced-reviews_val.csv'), index=False,
                                              index_label=["Score", "Text"])
