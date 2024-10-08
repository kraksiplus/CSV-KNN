#!/bin/bash

# Script to run main.py 10 times consecutively

for i in {1..10}
do
    echo "Running iteration $i"
    python3 main.py
done

