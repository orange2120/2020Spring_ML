#!/bin/bash

if [ $# != 1 ]; then
    echo "bash  hw3_train.sh <data directory>"
    exit 1
fi

python3 load_data.py $1
python3 train.py $1

echo "done."