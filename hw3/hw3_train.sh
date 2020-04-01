#!/bin/bash

if [ $# != 2 ]; then
    echo "bash  hw3_train.sh <data directory>\n"
    exit 1
fi

python3 train.py $1