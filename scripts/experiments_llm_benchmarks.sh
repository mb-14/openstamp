#!/bin/bash

set -e  # Optional: exit immediately if a command fails


# Change this to your local directory containing the watermaked model
models=(
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.3"
)


datasets=(boolq arc_challenge hellaswag)
# datasets=(arc_challenge hellaswag)

#* Test the performance of models on different benchmarks
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running model=$model on benchmark=$dataset"
        lm_eval --model hf --model_args "pretrained=${model},dtype=bfloat16" --tasks $dataset \
        --num_fewshot 5 \
        --batch_size 8 \
        --output_path "results/$dataset" --log_samples
    done
done