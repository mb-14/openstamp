#!/bin/bash

set -e  # Optional: exit immediately if a command fails

# models=(
# "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-greenlist-bias-k128-seed15485863" 
# "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-arcsine-noise-seed12997009" \
# "/pool.ssd/assets/models/mistralai/Mistral-7B-v0.3-watermarked-greenlist-bias-k128-seed12997009" 
# "/pool.ssd/assets/models/mistralai/Mistral-7B-v0.3-watermarked-arcsine-noise-seed22983996" \
# "meta-llama/Llama-2-7b-hf")
# "mistralai/Mistral-7B-v0.3") 

models=("meta-llama/Llama-2-7b-hf")
models=(
    "meta-llama/Llama-2-7b-hf" \
    "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-gaussmark-sigma0.04-seed15485863" \
    "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-gaussmark-sigma0.04-seed12997009" \
    "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-gaussmark-sigma0.04-seed22983996" \
    "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-greenlist-bias-k235-seed15485863" \
    "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-greenlist-bias-k235-seed12997009" \
    "/pool.ssd/assets/models/meta-llama/Llama-2-7b-hf-watermarked-greenlist-bias-k235-seed22983996" \
    "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2"
)

# models = ["mbakshi1094/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1.25"]
# models=("cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2")

datasets=(mmlu)
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