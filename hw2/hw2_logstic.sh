#!/bin/bash
if [ $# != 6 ]; then
    echo "Usage: bash hw2_logstic.sh [raw training data] [raw testing data] \n
    [preprocessed training feature] [training label]\n
    [preprocessed testing feature] [output path]"
    exit 1
fi

python3.7 traing_logstic.py $1 $2 $3 $4 $5 $6