#!/bin/bash

# Default parameter values
DELTA=3.0
GAMMA=0.5
NUM_SAMPLES=500
PARAPHRASE=0
SEED=15485863
watermark="mb"
generate=1
dataset="realnewslike"
model="meta-llama/Llama-2-7b-hf"
sigma=0.018
eval_ppl=1
target_param_name="lm_head.weight"
K=128
rl_model_path="/"
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --k)
    K="$2"
    shift
    ;;
  --delta)
    DELTA="$2"
    shift
    ;;
  --gamma)
    GAMMA="$2"
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
  --eval_ppl)
    eval_ppl="$2"
    shift
    ;;
  --seed)
    SEED="$2"
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
  --sigma)
    sigma="$2"
    shift
    ;;
  --distribution)
    distribution="$2"
    shift
    ;;
  --target_param_name)
    target_param_name="$2"
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
  output_dir="output/new/${model_suffix}"
fi

log_dir="${output_dir}/logs"
mkdir -p "$log_dir"

set -x

timestamp=$(date +"%Y%m%d_%H%M%S_%3N")
# if watermark is gaussmark, set the output file name accordingly
if [ "$watermark" == "gaussmark" ]; then
  output_file="${output_dir}/output_seed=${SEED}_sigma=${sigma}_watermark=${watermark}_dataset=${dataset}.json"
elif [ "$watermark" == "mb" ]; then
  output_file="${output_dir}/output_delta=${DELTA}_gamma=${GAMMA}_k=${K}_seed=${SEED}_watermark=${watermark}_dataset=${dataset}.json"
elif [ "$watermark" == "kgw" ] || [ "$watermark" == "kgw_llr" ]; then
  output_file="${output_dir}/output_seed=${SEED}_delta=${DELTA}_gamma=${GAMMA}_watermark=${watermark}_dataset=${dataset}.json"
elif [ "$watermark" == "rl" ]; then
  output_file="${output_dir}/output_watermark=${watermark}_dataset=${dataset}.json"
  base_dir="/pool.ssd/users/miroojin/watermarking_rl"
  rl_model_path="${base_dir}/c4_llama2-7b_llama2-1.1b_b4_step2500_dosample"
elif [ "$watermark" == "noise" ]; then
  output_file="${output_dir}/output_seed=${SEED}_distribution=${distribution}_delta=${DELTA}_watermark=${watermark}_dataset=${dataset}.json"
elif [ "$watermark" == "distilled" ]; then
  output_file="${output_dir}/output_watermark=${watermark}_dataset=${dataset}.json"
else
  echo "Unsupported watermark type ${watermark}."
  exit 1
fi

# if [ "$dataset" = "realnewslike" ]; then
#   dataset_args="--dataset_path allenai/c4 \
#     --dataset_config_name realnewslike \
#     --dataset_split validation \
#     --data_field text"
# elif [ "$dataset" = "wikipedia" ]; then
#   dataset_args="--dataset_path wikipedia \
#     --dataset_config_name 20220301.en \
#     --dataset_split train \
#     --data_field text \
#     --streaming"
# elif [ "$dataset" = "arxiv" ]; then
#   dataset_args="--dataset_path armanc/scientific_papers \
#     --dataset_config_name arxiv \
#     --dataset_split test \
#     --data_field article"
# elif [ "$dataset" = "booksum" ]; then
#   dataset_args="--dataset_path kmfoda/booksum \
#     --dataset_split test \
#     --data_field chapter"
# else
#   echo "Unsupported dataset ${dataset}."
#   exit 1
# fi

if [ "$generate" -eq 1 ]; then
  python -m scripts.generate_samples --num_samples $NUM_SAMPLES \
    --output_file $output_file \
    --dataset $dataset \
    --delta $DELTA \
    --gamma $GAMMA \
    --hash_key $SEED \
    --watermark $watermark \
    --model_name $model \
    --sigma $sigma \
    --k $K \
    --target_param_name $target_param_name \
    --distribution $distribution \
    --rl_model_path $rl_model_path &>"$log_dir/generate_${timestamp}.log"
fi

# Generate paraphrases if PARAPHRASE is set to 1
if [ "$PARAPHRASE" -eq 1 ]; then
  python scripts/paraphrase.py \
    --output_file $output_file \
    --lex 60 --order 0 &>"$log_dir/paraphrase_l60_${timestamp}.log"

  python scripts/paraphrase.py \
    --output_file $output_file \
    --lex 20 --order 0 &>"$log_dir/paraphrase_l20_${timestamp}.log"
fi

MODEL=$model K=$K OUTPUT_FILE=$output_file papermill notebooks/test_watermarking_v1.ipynb "$log_dir/tw_$timestamp.ipynb"

# Evaluate perplexity if eval_ppl is set to 1
if [ "$eval_ppl" -eq 1 ]; then
  python scripts/evaluate_ppl.py \
    --batch_size 16 \
    --output_file $output_file
fi
