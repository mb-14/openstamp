#!/bin/bash

export CUDA_VISIBLE_DEVICES='0,1,5,6'
base_selector_dir="saved_models_new"

# --- 1. Parse Arguments ---
watermark="openstamp"
target_config="0" # Default value
seed=15485863
# 0 - Lora on all internal linear layers
# 1 - Lora on all internal linear layers + unembedding layer
# 2 - Full fine-tuning (no LoRA) on unembedding layer only

while getopts "w:t:" opt; do
    case $opt in
        w) watermark="$OPTARG" ;;
        t) target_config="$OPTARG" ;;
        seed) seed="$OPTARG" ;;
        *) exit 1 ;;
    esac
done

# --- 2. Validation ---
case "$watermark" in
    "openstamp"|"gaussmark"|"kgw_distilled") echo "Watermark: $watermark" ;;
    *) echo "Error: Invalid watermark type."; exit 1 ;;
esac

case "$target_config" in
    0|1|2) echo "Target Param Config: $target_config" ;;
    *) echo "Error: target_param_config must be 0, 1, or 2"; exit 1 ;;
esac

# --- 3. Define the Training Function ---
run_train() {
    local base_name=$1
    local model_path=$2
    local selector_dir=$3
    local w_type=$4

    # Accessing the global $target_config to create suffixes
    local final_run_name="${base_name}_config${target_config}"

    unset WANDB_RUN_NAME WANDB_RUN_ID
    export WANDB_RUN_NAME="$final_run_name"
    
    echo "------------------------------------------------"
    echo "Running training for: ${WANDB_RUN_NAME}"
    echo "Target Config: ${target_config}"
    echo "------------------------------------------------"

    accelerate launch --config_file finetuning/train_z1.yaml -m finetuning.run_trainer_finetune \
        --model_name_or_path "$model_path" \
        --selector_matrix_dir "$selector_dir" \
        --watermark_type "$w_type" \
        --target_param_config "$target_config" \
        --run_name "$final_run_name" \
        --output_dir "finetuning/colm/$final_run_name" \
        --watermark_seed $seed \
        --max_steps 2500 \
        --num_train_epochs 1 \
        --dtype bfloat16 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_checkpointing false \
        --gradient_accumulation_steps 8 \
        --do_train true \
        --save_strategy steps \
        --save_steps 500 \
        --report_to wandb \
        --warmup_ratio 0.1 \
        --learning_rate 2e-5 \
        --dataset_num_proc 32 \
        --lr_scheduler_type cosine \
        --optim adafactor \
        --gpu_ids all
}

# --- 4. Execution Logic ---
if [ "$watermark" == "gaussmark" ]; then
    run_train "gaussmark_Llama-2-7b-hf" "meta-llama/Llama-2-7b-hf" "" "gaussmark"

elif [ "$watermark" == "kgw_distilled" ]; then
    run_train "kgw_distilled_Llama-2-7b-hf" "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2" "" "kgw_distilled"

elif [ "$watermark" == "openstamp" ]; then
    selector_matrices=(
        # "openwebtext_Llama-2-7b-hf_k256"
        "openwebtext_Llama-2-7b-hf_k254_semalign_contrastive_Qwen3-Embedding-8B"
        # "openwebtext_Llama-2-7b-hf_k256_semalign_ridge_Qwen3-Embedding-8B"
    )

    for sm in "${selector_matrices[@]}"; do
        # Extract name and full path
        r_name=$(basename "$sm")
        s_path="$base_selector_dir/$sm"
        
        run_train "$r_name" "meta-llama/Llama-2-7b-hf" "$s_path" "openstamp"
    done
fi