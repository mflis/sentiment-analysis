#!/usr/bin/env bash

cat data/AmazonReviews.csv  | awk '$1 <  3 && NR >= 1 && NR <= 16934' > negatives_val.csv
cat data/AmazonReviews.csv  | awk '$1 <  3 && NR > 16934 && NR <= 84674' > negatives_train.csv
cat data/AmazonReviews.csv  | awk '$1 >  3 && NR >= 1 && NR <= 16934' > positives_val.csv
cat data/AmazonReviews.csv  | awk '$1 >  3 && NR > 16934 && NR <= 84674' > positives_train.csv
echo "Score,Text" >> balanced-reviews_train.csv
echo "Score,Text" >> balanced-reviews_val.csv
paste -d"\n" positives_train.csv negatives_train.csv  | shuf  >> balanced-reviews_train.csv
paste -d"\n" positives_val.csv negatives_val.csv  | shuf  >> balanced-reviews_val.csv
rm  negatives*.csv positives*.csv
mv balanced-reviews_val.csv  balanced-reviews_train.csv data/


