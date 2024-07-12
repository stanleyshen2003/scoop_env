#!/bin/bash
for i in {0..200}; do
    echo "Running iteration $i"

    timeout 1800 python random_setting.py

    # Check the exit status of the Python script
    if [ $? -eq 124 ]; then
        echo "Timeout occurred for iteration $i"
    fi
done
