#!/bin/bash

if [ $# != 3 ]; then
    echo "bash  hw3_test.sh  <data directory>  <prediction file>\n"
    exit 1
fi

python3 test.py $1 $2