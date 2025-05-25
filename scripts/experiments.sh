#!/bin/bash

set -e  # Optional: exit immediately if a command fails

seeds=(15485863 12997009 22983996)
align=(0)
k=(4)
gamma=(0.3)
delta=(1.2)
models=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.3")
datasets=("realnewslike" "arxiv" "booksum" "wikipedia")
models=("meta-llama/Llama-2-7b-hf")
datasets=("realnewslike")

# Test watermarking with different parameters
for a in "${align[@]}"; do
  for k_val in "${k[@]}"; do
    for g in "${gamma[@]}"; do
      for d in "${delta[@]}"; do
        for dataset in "${datasets[@]}"; do
          for model in "${models[@]}"; do
            for seed in "${seeds[@]}"; do
              echo "Running test_watermarking with align=$a, k=$k_val, gamma=$g, delta=$d, dataset=$dataset"
              ./scripts/test_watermarking.sh --gamma "$g" --delta "$d" --k "$k_val" \
              --seed $seed --num_samples 500 --align "$a" --paraphrase 0 --train 1 --generate 1 \
              --dataset "$dataset" --watermark mb --model $model
            done
          done
        done
      done
    done
  done
done
