#!/bin/bash

set -e  # Optional: exit immediately if a command fails

# models=(
# "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-greenlist-bias-k128-seed15485863" 
# "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-arcsine-noise-seed12997009" \
# "/pool.ssd/assets/models/mistralai/Mistral-7B-v0.3-watermarked-greenlist-bias-k128-seed12997009" 
# "/pool.ssd/assets/models/mistralai/Mistral-7B-v0.3-watermarked-arcsine-noise-seed22983996" \
# "meta-llama/Llama-2-7b-hf")
# "mistralai/Mistral-7B-v0.3") 

models=("$@")
# datasets=(super-glue-lm-eval-v1)
datasets=(arc_challenge hellaswag)

#* Test the performance of models on different benchmarks
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running model=$model on benchmark=$dataset"
        accelerate launch --main_process_port 0 -m lm_eval --model hf --model_args "pretrained=${model},dtype=bfloat16" --tasks $dataset --batch_size 16 \
        --output_path "results/$dataset" --log_samples
    done
done