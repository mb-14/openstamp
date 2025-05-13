#!/bin/bash

set -e  # Optional: exit immediately if a command fails

seeds=(15485863 12997009 22983996)

sigma=(0.04)

#models=("mistralai/Mistral-7B-v0.3")
models=("meta-llama/Llama-2-7b-hf")
datasets=("realnewslike" "arxiv" "booksum" "wikipedia")

# Test watermarking with different parameters
for s in "${sigma[@]}"; do
  for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
      for model in "${models[@]}"; do
        echo "Running test_watermarking with sigma=$s, seed=$seed, dataset=$dataset, model=$model"
        ./scripts/test_watermarking.sh --seed "$seed" --num_samples 500 --paraphrase 0 --train 0 --generate 0 \
        --dataset "$dataset" --watermark gaussmark --model $model
      done
    done
  done
done