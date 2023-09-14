#!/bin/bash

echo "Starting script..."

for i in {5..6}
do
    threshold=$(awk "BEGIN { printf \"%.2f\", (50 + $i * 25) / 100 }")
    echo "$threshold"
    kungfu-run -np 4 -H 83.212.80.31:1,83.212.80.45:1,83.212.80.103:1,83.212.80.107:1 -nic eth1 python3 examples/fda_examples/tf2_mnist_naive_fda.py --epochs 30 -l --model lenet5 --threshold "$threshold"
    sleep 1m
    kungfu-run -np 4 -H 83.212.80.31:1,83.212.80.45:1,83.212.80.103:1,83.212.80.107:1 -nic eth1 python3 examples/fda_examples/tf2_mnist_naive_fda.py --epochs 30 -l --model lenet5 --threshold "$threshold"
    sleep 1m
done



