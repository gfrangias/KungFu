#!/bin/sh
 

for i in {1..15}
    do 
        kungfu-run -np 4 -strategy STAR -H 83.212.80.31:1,83.212.80.45:1,83.212.80.103:1,83.212.80.107:1 -nic eth1 python3 examples/fda_mnist_gradient_tape_star.py --threshold 1.3 --fda naive --batches 1000
    done

