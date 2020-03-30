#!/bin/bash
if [ $# != 2 ]; then
    echo "Usage: bash hw1_best.sh <input file> <output file>"
    exit 1
fi

echo "Input file: " $1
echo "Output file: " $2
python3 best_test.py $1 $2