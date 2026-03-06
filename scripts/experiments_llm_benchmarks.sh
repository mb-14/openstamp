#!/bin/bash

set -e  # Optional: exit immediately if a command fails

models=(
    "/data/users/miroojin/saksham/watermark-adapters/output/watermarked_models/meta-llama/Llama-2-7b-hf-watermarked-openstamp-semalign-contrastive-Qwen3-Embedding-8B-k254-delta1.0-gamma-0.25-seed22983996" \
    "/data/users/miroojin/saksham/watermark-adapters/output/watermarked_models/meta-llama/Llama-2-7b-hf-watermarked-openstamp-semalign-contrastive-Qwen3-Embedding-8B-k254-delta1.0-gamma-0.25-seed15485863" \
    "/data/users/miroojin/saksham/watermark-adapters/output/watermarked_models/meta-llama/Llama-2-7b-hf-watermarked-openstamp-semalign-contrastive-Qwen3-Embedding-8B-k254-delta1.0-gamma-0.25-seed12997009" \
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