#!/bin/bash

if [ $# != 2 ]; then
    echo "bash  hw4_test.sh <testing data> <prediction file>"
    exit 1
fi

# download model command ...
python3 main_test.py $1 $2

echo "done."