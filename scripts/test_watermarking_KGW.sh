#!/bin/bash

# Default parameter values
GAMMA=0.5
DELTA=2.0
PREFIX_LENGTH=1 
NUM_SAMPLES=500
PARAPHRASE=0
SEED=15485863
watermark="KGW"
generate=1
train=0
dataset="realnewslike"
model="meta-llama/Llama-2-7b-hf"
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --gamma)
    GAMMA="$2"
    shift
    ;;
  --delta)
    DELTA="$2"
    shift
    ;;
  --prefix_length)
    PREFIX_LENGTH="$2"
    shift
    ;;
  --seed)
    SEED="$2"
    shift
    ;;
  --num_samples)
    NUM_SAMPLES="$2"
    shift
    ;;
  --paraphrase)
    PARAPHRASE="$2"
    shift
    ;;
  --generate)
    generate="$2"
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
  output_dir="output/${model_suffix}"
fi

log_dir="${output_dir}/logs"
mkdir -p "$log_dir"

timestamp=$(date +"%Y%m%d_%H%M%S")
# if watermark is gaussmark, set the output file name accordingly
if [ "$watermark" == "KGW" ]; then
  output_file="${output_dir}/output_prefix=${PREFIX_LENGTH}_delta=${DELTA}_gamma=${GAMMA}_seed=${SEED}_watermark=${watermark}_dataset=${dataset}.json"
fi

if [ "$dataset" = "realnewslike" ]; then
    dataset_args="--dataset_path allenai/c4 \
    --dataset_config_name realnewslike \
    --dataset_split validation \
    --data_field text"
elif [ "$dataset" = "wikipedia" ]; then
    dataset_args="--dataset_path wikipedia \
    --dataset_config_name 20220301.en \
    --dataset_split train \
    --data_field text"
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
  python scripts/generate_samples_vllm_KGW.py --num_samples $NUM_SAMPLES \
  --output_file $output_file \
  ${dataset_args} \
  --watermark $watermark \
  --gamma $GAMMA \
  --delta $DELTA \
  --hash_key $SEED \
  --prefix_length $PREFIX_LENGTH \
  --model_name $model 
fi

# Generate paraphrases if PARAPHRASE is set to 1
if [ "$PARAPHRASE" -eq 1 ]; then
  python scripts/paraphrase.py \
    --output_file $output_file \
    --lex 60 --order 0

  python scripts/paraphrase.py \
    --output_file $output_file \
    --lex 20 --order 0
fi

python scripts/evaluate_ppl.py \
  --batch_size 16 \
  --output_file $output_file

ROOT_PATH="" MODEL=$model OUTPUT_FILE=$output_file papermill notebooks/test_watermarking_KGW.ipynb "$log_dir/tw_$timestamp.ipynb"