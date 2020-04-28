#!/bin/bash

if [ $# != 2 ]; then
    echo "bash  hw3_test.sh <data directory> <prediction file>"
    exit 1
fi

python3 test.py $1 $2

echo "done."