#!/bin/bash

set -e # Optional: exit immediately if a command fails

datasets=("realnewslike" "arxiv" "booksum" "wikipedia")
models=("meta-llama/Llama-2-7b-hf")
models=("cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2")
datasets=("realnewslike")
steps=(0 500 1000 1500 2000 2500 3000 3500 4000)
checkpoint_dirs=("kgw_distilled")
watermark=("distilled")

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for step in "${steps[@]}"; do
      for wm in "${watermark[@]}"; do
        for checkpoint_dir in "${checkpoint_dirs[@]}"; do
          echo "Running with dataset: $dataset, model: $model, step: $step, watermark: $wm"
          ./scripts/test_watermarking_ft.sh --num_samples 500 --generate 1 \
            --dataset "$dataset" --watermark $wm --model $model --step $step --checkpoint_dir $checkpoint_dir
        done
      done
    done
  done
done
