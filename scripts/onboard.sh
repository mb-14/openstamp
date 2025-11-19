#!/bin/bash

# Usage: ./onboard.sh [MODEL_ID]
MODEL_ID=${1:-"meta-llama/Llama-3.1-70B"}
MODEL_NAME_ONLY=${MODEL_ID##*/}

echo "=================================================="
echo "Starting workflow for Model: $MODEL_ID"
echo "Dataset path target: data/openwebtext_${MODEL_NAME_ONLY}"
echo "=================================================="

echo "[1/4] Downloading model..."
hf download "$MODEL_ID" || { echo "Error: Model download failed."; exit 1; }

echo "[2/4] Preprocessing..."
python -m scripts.preprocess --model-name "$MODEL_ID" --total 1000 || { echo "Error: Preprocessing failed."; exit 1; }

echo "[3/4] Generating hidden states..."
python scripts/generate_hidden_states.py \
    --model "$MODEL_ID" \
    --dataset_path "data/openwebtext_${MODEL_NAME_ONLY}" \
    --total_samples 1500000 || { echo "Error: Hidden state generation failed."; exit 1; }

echo "[4/4] Training selector..."
python -m scripts.train_selector --k 128 --model-name "$MODEL_ID" || { echo "Error: Training selector failed."; exit 1; }

echo "=================================================="
echo "Workflow completed successfully for $MODEL_ID!"
echo "=================================================="