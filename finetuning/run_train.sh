#!/bin/bash

base_selector_dir="saved_models_new"

# --- 1. Parse Arguments ---
watermark="openstamp"
model="llama"
target_config="1" # Default value (config 1: Lora on all internal linear layers + unembedding layer)
seed=15485863
distilled_model_path="cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2"
# 0 - Lora on all internal linear layers
# 1 - Lora on all internal linear layers + unembedding layer
# 2 - Full fine-tuning (no LoRA) on unembedding layer only

base_checkpoint_dir="finetuning/colm"
output_dir=""

usage() {
    echo "Usage: $0 [--watermark value] [--model value] [--seed value] [--distilled_model_path value] [--output_dir value]"
}

set_option() {
    local option="$1"
    local value="$2"

    case "$option" in
        --watermark) watermark="$value" ;;
        --model) model="$value" ;;
        --seed) seed="$value" ;;
        --distilled_model_path) distilled_model_path="$value" ;;
        --output_dir) output_dir="$value" ;;
        *)
            echo "Error: Unknown option '$option'"
            usage
            exit 1
            ;;
    esac
}

require_value() {
    if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: Missing value for $1"
        usage
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watermark|--model|--seed|--distilled_model_path|--output_dir)
            require_value "$1" "$2"
            set_option "$1" "$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        -*)
            usage
            exit 1
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            usage
            exit 1
            ;;
    esac
done

# --- 2. Validation ---
case "$watermark" in
    "openstamp"|"gaussmark"|"christ"|"kgw_distilled") echo "Watermark: $watermark" ;;
    *) echo "Error: Invalid watermark type."; exit 1 ;;
esac

case "$model" in
    "llama"|"mistral"|"qwen") echo "Model: $model" ;;
    *) echo "Error: model must be 'llama', 'mistral', or 'qwen'"; exit 1 ;;
esac

# --- 3. Define the Training Function ---
run_train() {
    local base_name=$1
    local model_path=$2
    local selector_dir=$3
    local w_type=$4

    # Accessing the global $target_config to create suffixes
    local final_run_name="${base_name}_config${target_config}_seed${seed}"
    local final_output_dir="${base_checkpoint_dir}/${final_run_name}"

    if [[ -n "$output_dir" ]]; then
        final_output_dir="$output_dir"
    fi

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
        --output_dir "$final_output_dir" \
        --watermark_seed $seed \
        --max_steps 2500 \
        --num_train_epochs 1 \
        --dtype bfloat16 \
        --per_device_train_batch_size 12 \
        --per_device_eval_batch_size 12 \
        --gradient_checkpointing false \
        --gradient_accumulation_steps 4 \
        --do_train true \
        --save_strategy steps \
        --save_steps 500 \
        --report_to wandb \
        --warmup_ratio 0.1 \
        --learning_rate 2e-5 \
        --dataset_num_proc 32 \
        --lr_scheduler_type cosine \
        --optim adafactor \
        --loss_type nll \
        --gpu_ids all
    
    echo "Checkpoint saved to: $final_output_dir"
}

# --- 4. Execution Logic ---
# Set model path based on model choice
if [ "$model" == "llama" ]; then
    model_path="meta-llama/Llama-2-7b-hf"
    model_suffix="Llama-2-7b-hf"
elif [ "$model" == "mistral" ]; then
    model_path="mistralai/Mistral-7B-v0.3"
    model_suffix="Mistral-7B-v0.3"
elif [ "$model" == "qwen" ]; then
    model_path="Qwen/Qwen2.5-7B"
    model_suffix="Qwen2.5-7B"
fi

if [ "$watermark" == "gaussmark" ]; then
    run_train "gaussmark_${model_suffix}" "$model_path" "" "gaussmark"

elif [ "$watermark" == "christ" ]; then
    run_train "christ_${model_suffix}" "$model_path" "" "christ"

elif [ "$watermark" == "kgw_distilled" ]; then
    if [ "$model" != "llama" ]; then
        echo "Error: kgw_distilled is only supported for model=llama"
        exit 1
    fi

    # Change this path with --distilled_model_path if needed
    model_name_or_path="$distilled_model_path"
    # Run name is a label for wandb and output directory
    run_name="kgw_distilled_Llama-2-7b-hf"

    run_train "$run_name" "$model_name_or_path" "" "kgw_distilled"

elif [ "$watermark" == "openstamp" ]; then
    if [ "$model" == "llama" ]; then
        selector_matrices=(
            "openwebtext_Llama-2-7b-hf_k254_semalign_contrastive_Qwen3-Embedding-8B"
        )
    elif [ "$model" == "mistral" ]; then
        selector_matrices=(
            "openwebtext_Mistral-7B-v0.3_k255"
        )
    elif [ "$model" == "qwen" ]; then
        selector_matrices=(
            "openwebtext_Qwen2.5-7B_k251_semalign_contrastive_Qwen3-Embedding-8B"
            "openwebtext_Qwen2.5-7B_k256"
        )
    fi

    for sm in "${selector_matrices[@]}"; do
        r_name=$(basename "$sm")
        s_path="$base_selector_dir/$sm"

        run_train "$r_name" "$model_path" "$s_path" "openstamp"
    done
fi