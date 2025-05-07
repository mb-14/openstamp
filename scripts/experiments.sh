#!/bin/bash

set -e  # Optional: exit immediately if a command fails

seeds=(15485863 12997009 22983996)
align=(1)
k=(4)
gamma=(0.4)
delta=(1.5)

# Test watermarking with different parameters
for a in "${align[@]}"; do
  for k_val in "${k[@]}"; do
    for g in "${gamma[@]}"; do
      for d in "${delta[@]}"; do
        for seed in "${seeds[@]}"; do
          echo "Running test_watermarking with align=$a, k=$k_val, gamma=$g, delta=$d, seed=$seed"
          ./scripts/test_watermarking.sh --gamma "$g" --delta "$d" --k "$k_val" \
          --seed "$seed" --num_samples 500 --align "$a" --paraphrase 0 --train 1 --generate 1 \
          --dataset booksum --watermark mb
        done
      done
    done
  done
done
