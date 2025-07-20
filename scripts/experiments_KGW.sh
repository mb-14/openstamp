#!/bin/bash

set -e  # Optional: exit immediately if a command fails

# Check if a model was passed in
# if [ -z "$1" ]; then
#     echo "Usage: $0 <model_name>"
#     echo "Example: $0 meta-llama/Llama-2-7b-hf"
#     exit 1
# fi

model_arg="$1"
models=("$model_arg")  # Array with just the passed model

seeds=(15485863 12997009 22983996)
# gamma=(0.25 0.3 0.5)
# delta=(0.0 1.0 2.0 3.0)
# prefix_length=(0 1)
# models=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.3")
# datasets=("wikipedia")
# datasets=("realnewslike" "arxiv" "booksum" "wikipedia")
# prefix_length=(0)

#! For the paraphrasing
# models=("/merged_models/mistral_greenlist")
# seeds=(12997009)
datasets=("realnewslike")
gamma=(0.25)
delta=(1.0)
prefix_length=(0 1)

# Test watermarking with different parameters
for p in "${prefix_length[@]}"; do
    for g in "${gamma[@]}"; do
        for d in "${delta[@]}"; do
            for dataset in "${datasets[@]}"; do
                for model in "${models[@]}"; do
                    for seed in "${seeds[@]}"; do
                        echo "Running test_watermarking_KGW on model=$model with prefix_length=$p, gamma=$g, delta=$d, seed=$seed, dataset=$dataset"
                        ./scripts/test_watermarking_KGW.sh \
                        --gamma "$g" --delta "$d" --prefix_length "$p" \
                        --seed $seed --num_samples 500 --paraphrase 0 --generate 1 \
                        --dataset "$dataset" --model $model
                    done
                done
            done
        done
    done
done