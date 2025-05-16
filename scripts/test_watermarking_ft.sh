#!/bin/bash


NUM_SAMPLES=500
watermark="mb"
generate=1
dataset="realnewslike"
model="meta-llama/Llama-2-7b-hf"
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --num_samples)
    NUM_SAMPLES="$2"
    shift
    ;;
  --step)
    step="$2"
    shift
    ;;
  --output_dir)
    output_dir="$2"
    shift
    ;;
  --watermark)
    watermark="$2"
    shift
    ;;
  --generate)
    generate="$2"
    shift
    ;;
  --dataset)
    dataset="$2"
    shift
    ;;
  --model)
    model="$2"
    shift
    ;;
  *)
    echo "Unknown parameter passed: $1"
    exit 1
    ;;
  esac
  shift
done

# Check if output_dir is set, if not use default
model_suffix="${model#*/}"

if [ -z "$output_dir" ]; then
  output_dir="output/${model_suffix}/lora"
fi

log_dir="${output_dir}/logs"
mkdir -p "$log_dir"

output_file="${output_dir}/output_watermark=${watermark}_dataset=${dataset}_step=${step}.json"


if [ "$dataset" = "realnewslike" ]; then
    dataset_args="--dataset_path allenai/c4 \
    --dataset_config_name realnewslike \
    --dataset_split validation \
    --data_field text"
elif [ "$dataset" = "wikipedia" ]; then
    dataset_args="--dataset_path wikipedia \
    --dataset_config_name 20220301.en \
    --dataset_split train \
    --data_field text \
    --streaming"
elif [ "$dataset" = "arxiv" ]; then
    dataset_args="--dataset_path armanc/scientific_papers \
    --dataset_config_name arxiv \
    --dataset_split test \
    --data_field article"
elif [ "$dataset" = "booksum" ]; then
    dataset_args="--dataset_path kmfoda/booksum \
    --dataset_split test \
    --data_field chapter"
else
    echo "Unsupported dataset ${dataset}."
    exit 1
fi


if [ "$generate" -eq 1 ]; then
  python -m scripts.generate_samples_ft --num_samples $NUM_SAMPLES \
  --output_file $output_file \
  ${dataset_args} \
  --watermark $watermark \
  --model_name $model \
  --step $step
fi


MODEL=$model OUTPUT_FILE=$output_file papermill notebooks/test_watermarking_v1.ipynb "$log_dir/tw_$timestamp.ipynb"

python scripts/evaluate_ppl.py \
  --batch_size 16 \
  --output_file $output_file