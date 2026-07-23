#!/bin/bash

base_selector_dir="saved_models_new"

# --- 1. Parse Arguments ---
watermark="openstamp"
model="llama"
target_config="1" # Default value (config 1: Lora on all internal linear layers + unembedding layer)
seed=15485863
distilled_model_path=""
ft_dataset="openwebtext"
# 0 - Lora on all internal linear layers
# 1 - Lora on all internal linear layers + unembedding layer
# 2 - Full fine-tuning (no LoRA) on unembedding layer only

base_checkpoint_dir="finetuning/colm"
output_dir=""

usage() {
    echo "Usage: $0 [--watermark value] [--model value] [--seed value] [--ft_dataset openwebtext|alpaca] [--distilled_model_path value] [--output_dir value]"
}

set_option() {
    local option="$1"
    local value="$2"

    case "$option" in
        --watermark) watermark="$value" ;;
        --model) model="$value" ;;
        --seed) seed="$value" ;;
        --ft_dataset) ft_dataset="$value" ;;
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
        --watermark|--model|--seed|--ft_dataset|--distilled_model_path|--output_dir)
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
    "openstamp"|"gaussmark"|"unremovable"|"christ"|"kgw_distilled") echo "Watermark: $watermark" ;;
    *) echo "Error: Invalid watermark type."; exit 1 ;;
esac

case "$model" in
    "llama"|"mistral"|"qwen") echo "Model: $model" ;;
    *) echo "Error: model must be 'llama', 'mistral', or 'qwen'"; exit 1 ;;
esac

case "$ft_dataset" in
    "openwebtext"|"alpaca") echo "FT dataset: $ft_dataset" ;;
    *) echo "Error: ft_dataset must be 'openwebtext' or 'alpaca'"; exit 1 ;;
esac

if [[ "$ft_dataset" == "alpaca" ]]; then
    base_checkpoint_dir="finetuning/colm_alpaca"
fi

# --- 3. Define the Training Function ---
run_train() {
    local base_name=$1
    local model_path=$2
    local selector_dir=$3
    local w_type=$4

    # Accessing the global $target_config to create suffixes
    local final_run_name="${base_name}_config${target_config}_seed${seed}"
    if [[ "$ft_dataset" == "alpaca" ]]; then
        final_run_name="${base_name}_alpaca_config${target_config}_seed${seed}"
    fi
    local final_output_dir="${base_checkpoint_dir}/${final_run_name}"

    if [[ -n "$output_dir" ]]; then
        final_output_dir="$output_dir"
    fi

    unset WANDB_RUN_NAME WANDB_RUN_ID
    export WANDB_RUN_NAME="$final_run_name"

    echo "------------------------------------------------"
    echo "Running training for: ${WANDB_RUN_NAME}"
    echo "Target Config: ${target_config}"
    echo "FT dataset: ${ft_dataset}"
    echo "------------------------------------------------"

    local dataset_args=(--ft_dataset "$ft_dataset")
    if [[ "$ft_dataset" == "alpaca" ]]; then
        dataset_args+=(--max_length 512)
    fi

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
        --attn_implementation sdpa \
        --bf16 true \
        --per_device_train_batch_size 12 \
        --per_device_eval_batch_size 12 \
        --dataloader_num_workers 4 \
        --dataloader_pin_memory true \
        --gradient_checkpointing false \
        --gradient_accumulation_steps 4 \
        --do_train true \
        --save_strategy steps \
        --save_steps 500 \
        --report_to wandb \
        --warmup_ratio 0.1 \
        --learning_rate 2e-5 \
        --dataset_num_proc 8 \
        --lr_scheduler_type cosine \
        --optim adafactor \
        --loss_type nll \
        --gpu_ids all \
        "${dataset_args[@]}"
    
    echo "Checkpoint saved to: $final_output_dir"
}

resolve_distilled_path() {
    local base_model=$1
    local wm_seed=$2
    python - <<PY
from src.kgw_distilled import resolve_distilled_model
print(resolve_distilled_model("$base_model", $wm_seed))
PY
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

elif [ "$watermark" == "unremovable" ] || [ "$watermark" == "christ" ]; then
    run_train "unremovable_${model_suffix}" "$model_path" "" "unremovable"

elif [ "$watermark" == "kgw_distilled" ]; then
    if [[ "$model" != "llama" && "$model" != "mistral" ]]; then
        echo "Error: kgw_distilled is only supported for model=llama|mistral"
        exit 1
    fi

    if [[ -n "$distilled_model_path" ]]; then
        model_name_or_path="$distilled_model_path"
    else
        model_name_or_path="$(resolve_distilled_path "$model_path" "$seed")"
    fi
    echo "Distilled checkpoint: $model_name_or_path"
    run_name="kgw_distilled_${model_suffix}"

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