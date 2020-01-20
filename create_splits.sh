#!/bin/sh
set -e

usage() {
    echo "Create train/test splits for the dataset"
    echo "\nUsage: create_splits.sh INPUT_CSV_FILE OUTPUT_DIR"
    echo "\nCreates 3 output files in OUTPUT_DIR:"
    echo "\"train_top20.csv\": Dataset containing top 20 classes from even years"
    echo "\"train_other.csv\": Dataset containing all other classes from even years"
    echo "\"test.csv\": Test set containing items from odd years"
}

if [ ! $# -ge 2 ]
then
    usage
    exit 1
fi

CSV_IN=$1
OUT_DIR=$2

[ ! -f $CSV_IN  ] && echo "Input CSV $CSV_IN does not exist" && exit 1
[ ! -d $OUT_DIR  ] && echo "Output directory $OUT_DIR does not exist" && exit 1
[ ! -f ./top_20_classes.txt ] && echo "Top 20 classes file (top_20_classes.txt) missing" && exit 1

HEADER=$(head -n 1 $CSV_IN)
echo "$HEADER" > $OUT_DIR/test.csv

tail -n +2 $CSV_IN | awk -F ',' '{ print $5 }' | sort | uniq > $OUT_DIR/classes.txt

tail -n +2 $CSV_IN | awk -F ',' '($8 %2 == 0)' > $OUT_DIR/train.csv
tail -n +2 $CSV_IN | awk -F ',' '($8 %2 == 1)' >> $OUT_DIR/test.csv

echo "$HEADER" > $OUT_DIR/train_top20.csv
echo "$HEADER" > $OUT_DIR/train_other.csv
grep -f ./top_20_classes.txt -- $OUT_DIR/train.csv >> $OUT_DIR/train_top20.csv
grep -f ./top_20_classes.txt -v -- $OUT_DIR/train.csv >> $OUT_DIR/train_other.csv

rm $OUT_DIR/train.csv

echo "Done"
