#!/bin/bash

lower_limit=100
upper_limit=1000000000

n=$lower_limit
increment=100
next_increment=$((increment * 10))

while [ $n -le $upper_limit ]; do
    echo $n
    if [ $n -eq $next_increment ]; then
        increment=$next_increment
        next_increment=$((increment * 10))
    fi
    echo "Running for size: $n heads: 10"
    ./run_transformer_$1 $n 10 2>> results/flops_$1.txt
    for ((head = 25; head <= 100; head += 25)); do
        echo "Running for size: $n heads: $head"
        ./run_transformer_$1 $n $head 2>> results/flops_$1.txt
    done
    n=$((n + increment))
done