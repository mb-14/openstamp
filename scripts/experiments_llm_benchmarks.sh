#!/bin/bash 

set -e  # Optional: exit immediately if a command fails 

models=(
    "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-mb-seed12997009" "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-mb2-seed22983996" "mistralai/Mistral-7B-v0.3" \
"/pool.ssd/assets/models/mistralai/Mistral-7B-v0.3-watermarked-mb-seed12997009" "/pool.ssd/assets/models/mistralai/Mistral-7B-v0.3-watermarked-mb2-seed15485863" "meta-llama/Llama-2-7b-hf"
) 
datasets=(arc_challenge hellaswag gsm8k) 

#* Test the performance of models on different benchmarks 
for dataset in "${datasets[@]}"; do 
    for model in "${models[@]}"; do 
        echo "Running model=$model on benchmark=$dataset" 
        accelerate launch -m lm_eval --model hf --model_args "pretrained=${model},dtype=bfloat16" --tasks $dataset --batch_size 8 \
        --output_path "results/$dataset" --log_samples 
    done 
done