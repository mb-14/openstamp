#!/bin/bash

# Usage: ./onboard.sh [MODEL_ID]
MODEL_ID=${1:-"meta-llama/Llama-3.1-70B"}
MODEL_NAME_ONLY=${MODEL_ID##*/}

SEEDS=(42 1337)
DATASETS=("Skylion007/openwebtext" "HuggingFaceFW/fineweb")

echo "=================================================="
echo "Starting workflow for Model: $MODEL_ID"
echo "Dataset inputs: ${DATASETS[@]}"
echo "Output path template: data/<dataset_suffix>_${MODEL_NAME_ONLY}_seed<seed>/prefixes.pt"
echo "=================================================="

echo "[1/4] Downloading model..."
hf download "$MODEL_ID" || { echo "Error: Model download failed."; exit 1; }

echo "[2/4] Preprocessing..."
for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        DATASET_NAME_ONLY=${dataset##*/}
        OUTPUT_DIR="data/${DATASET_NAME_ONLY}_${MODEL_NAME_ONLY}_seed${seed}"
        echo "-> Preprocessing with seed=$seed dataset=$dataset (output: ${OUTPUT_DIR}/prefixes.pt)"
        python -m scripts.preprocess \
            --model-name "$MODEL_ID" \
            --seed "$seed" \
            --dataset-name "$dataset" \
            --total 1000 || { echo "Error: Preprocessing failed for seed=$seed dataset=$dataset."; exit 1; }
        
        # Generate hidden states from the prefixes saved by preprocess.py
        echo "-> Generating hidden states for seed=$seed dataset=$dataset"
        python scripts/generate_hidden_states.py \
            --dataset_path "$OUTPUT_DIR" \
            --model "$MODEL_ID" \
            --seed "$seed" \
            --total_samples 1500000 || { echo "Error: Hidden state generation failed for seed=$seed dataset=$dataset."; exit 1; }
        
        # Train selector using the hidden states for this seed/dataset
        echo "-> Training selector for seed=$seed dataset=$dataset"
        python -m scripts.train_selector \
            --dataset-name "$dataset" \
            --model-name "$MODEL_ID" \
            --seed "$seed" \
            --prf-key "$seed" \
            --k 128 || { echo "Error: Training selector failed for seed=$seed dataset=$dataset."; exit 1; }
    done
done

# echo "[4/4] Training selector..."
# python -m scripts.train_selector --k 128 --model-name "$MODEL_ID" || { echo "Error: Training selector failed."; exit 1; }

# echo "=================================================="
# echo "Workflow completed successfully for $MODEL_ID!"
# echo "=================================================="