#!/bin/bash

if [ $# != 2 ]; then
    echo "bash hw6_fgsm.sh <input dir> <output img dir>"
    exit 1
fi

python3 main.py $1 $2