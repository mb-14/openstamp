#!/bin/bash

set -e  # Optional: exit immediately if a command fails

seeds=(15485863 12997009 22983996)
gamma=(0.25 0.3 0.5)
delta=(0.0 1.0 2.0 3.0)
prefix_length=(0 1)
models=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.3")
datasets=("realnewslike" "arxiv" "booksum" "wikipedia")

#! For the paraphrasing experiments
seeds=(15485863)
gamma=(0.25 0.5)
delta=(0.0 2.0)
prefix_length=(0)
datasets=("realnewslike")

# Test watermarking with different parameters
for p in "${prefix_length[@]}"; do
    for g in "${gamma[@]}"; do
        for d in "${delta[@]}"; do
            for dataset in "${datasets[@]}"; do
                for model in "${models[@]}"; do
                    for seed in "${seeds[@]}"; do
                        echo "Running test_watermarking_KGW on model=$model with prefix_length=$p, gamma=$g, delta=$d, seed=$seed, dataset=$dataset"
                        ./scripts/test_watermarking_kgw.sh --gamma "$g" --delta "$d" --prefix_length "$p" \
                        --seed $seed --num_samples 500 --paraphrase 0 --generate 1 \
                        --dataset "$dataset" --model $model
                    done
                done
            done
        done
    done
done
