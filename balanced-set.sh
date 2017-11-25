#!/usr/bin/env bash

cat data/AmazonReviews.csv  | awk '$1 <  3' | head -n 84674 > negatives.csv
cat data/AmazonReviews.csv  | awk '$1 >  3' | head -n 84674 > positives.csv
echo "Score,Text" >> balanced-reviews.csv
paste -d"\n" positives.csv negatives.csv    | shuf  >> balanced-reviews.csv
rm  negatives.csv positives.csv
mv balanced-reviews.csv data