#!/bin/bash

# Usage: ./onboard.sh [MODEL_ID] [DATASET_PATH]
MODEL_ID=${1:-"meta-llama/Llama-3.1-70B"}
MODEL_NAME_ONLY=${MODEL_ID##*/}
DEFAULT_DATASET_PATH="data/openwebtext_${MODEL_NAME_ONLY}"
DATASET_PATH=${2:-"$DEFAULT_DATASET_PATH"}
set -x 

echo "=================================================="
echo "Starting workflow for Model: $MODEL_ID"
echo "Dataset path target: $DATASET_PATH"
echo "=================================================="

echo "[1/4] Downloading model..."
hf download "$MODEL_ID" || { echo "Error: Model download failed."; exit 1; }

echo "[2/4] Preprocessing..."
# python -m scripts.preprocess --model-name "$MODEL_ID" --total 1000 || { echo "Error: Preprocessing failed."; exit 1; }

echo "[3/4] Generating hidden states..."
python -m scripts.generate_hidden_states \
    --model "$MODEL_ID" \
    --dataset_path "$DATASET_PATH" \
    --total_samples 1500000 || { echo "Error: Hidden state generation failed."; exit 1; }

echo "[4/4] Training selector..."
python -m scripts.train_selector --k 256 --model-name "$MODEL_ID" || { echo "Error: Training selector failed."; exit 1; }

echo "=================================================="
echo "Workflow completed successfully for $MODEL_ID!"
echo "=================================================="