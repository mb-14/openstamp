#!/bin/bash

set -e  # Optional: exit immediately if a command fails

seeds=(15485863 12997009 22983996)

sigma=(0.04)

# Test watermarking with different parameters
for s in "${sigma[@]}"; do
  for seed in "${seeds[@]}"; do
    echo "Running test_watermarking with sigma=$s, seed=$seed"
    ./scripts/test_watermarking.sh --watermark gaussmark --seed "$seed" --num_samples 500 --paraphrase 0 --generate 1 --dataset booksum
  done
done