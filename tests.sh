#!/bin/bash

echo "Starting script..."

for i in {0..4}
do
    threshold=$(awk "BEGIN { printf \"%.2f\", (50 + $i * 25) / 100 }")
    echo "$threshold"
    kungfu-run -np 2 python3 examples/fda_examples/tf2_mnist_naive_fda.py --epochs 30 -l --model lenet5 --threshold "$threshold"
    sleep 1m
    kungfu-run -np 2 python3 examples/fda_examples/tf2_mnist_naive_fda.py --epochs 30 -l --model lenet5 --threshold "$threshold"
done
