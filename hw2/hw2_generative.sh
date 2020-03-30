#!/bin/bash
if [ $# != 6 ]; then
    echo "Usage: bash hw2_generative.sh [raw training data  (train.csv) ] [raw testing data (test_no_label.csv) ] \n
    [preprocessed training feature (X_train) ] [training label (Y_train)]\n
    [preprocessed testing feature (X_test) ] [output path]"
    exit 1
fi

python3 traing_gn.py $3 $4 $5 $6
python3 test_gn.py $5 $6