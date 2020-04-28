#!/bin/bash

if [ $# != 2 ]; then
    echo "bash  hw4_train.sh <training label data> <training unlabel data>"
    exit 1
fi

if [ ! -d "./data" ]; then
    mkdir -p ./data
    mkdir -p ./data/model
fi

python3 main_preprocess.py $1 $2
python3 main_train.py