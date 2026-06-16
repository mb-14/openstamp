#!/bin/bash

# Default parameter values
DELTA=3.0
GAMMA=0.5
NUM_SAMPLES=500
PARAPHRASE=0
SEED=15485863
watermark="openstamp"
generate=1
dataset="realnewslike"
model="meta-llama/Llama-2-7b-hf"
sigma=0.018
eval_ppl=1
target_param_name="lm_head.weight"
rl_model_path="."
step=0
checkpoint_dir="."
selector_matrix_dir=""
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
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
  --rl_model_path)
    rl_model_path="$2"
    shift
    ;;
  --checkpoint_dir)
    checkpoint_dir="$2"
    shift
    ;;
  --step)
    step="$2"
    shift
    ;;
  --selector_matrix_dir)
    selector_matrix_dir="$2"
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

set -ex

timestamp=$(date +"%Y%m%d_%H%M%S_%3N")
# if watermark is gaussmark, set the output file name accordingly
if [ "$watermark" == "gaussmark" ]; then
  output_file="${output_dir}/output_seed=${SEED}_sigma=${sigma}_watermark=${watermark}_dataset=${dataset}"
elif [ "$watermark" == "openstamp" ] || [ "$watermark" == "openstamp_binom" ] || [ "$watermark" == "openstamp_discrete" ]; then
  if [ -z "$selector_matrix_dir" ]; then
    echo "For OpenStamp watermarking, --selector_matrix_dir is required."
    exit 1
  fi
  selector_metrics_path="${selector_matrix_dir%/}/selector_metrics.json"
  k=""
  sem_align="false"
  embedding_model=""
  align_method=""
  if [ -f "$selector_metrics_path" ]; then
    k=$(jq -r '.k // empty' "$selector_metrics_path")
    sem_align=$(jq -r '.sem_align // false' "$selector_metrics_path")
    embedding_model=$(jq -r '.embedding_model // empty' "$selector_metrics_path")
    align_method=$(jq -r '.align_method // empty' "$selector_metrics_path")
  fi
  if [ -z "$k" ]; then
    echo "Missing k in selector metrics JSON: $selector_metrics_path"
    exit 1
  fi

  suffix=""
  if [ "$sem_align" = "true" ] && [ -n "$embedding_model" ] && [ -n "$align_method" ]; then
    suffix="_semalign_${align_method}_embedding=${embedding_model}"
  fi

  output_file="${output_dir}/output_delta=${DELTA}_gamma=${GAMMA}_k=${k}_seed=${SEED}_watermark=${watermark}_dataset=${dataset}${suffix}"
  K="$k"
elif [ "$watermark" == "kgw" ] || [ "$watermark" == "kgw_llr" ] || [ "$watermark" == "unigram" ]; then
  output_file="${output_dir}/output_seed=${SEED}_delta=${DELTA}_gamma=${GAMMA}_watermark=${watermark}_dataset=${dataset}"
elif [ "$watermark" == "rl" ] || [ "$watermark" == "distilled" ]; then
  output_file="${output_dir}/output_watermark=${watermark}_dataset=${dataset}"
elif [ "$watermark" == "noise" ]; then
  output_file="${output_dir}/output_seed=${SEED}_distribution=${distribution}_delta=${DELTA}_watermark=${watermark}_dataset=${dataset}"
else
  echo "Unsupported watermark type ${watermark}."
  exit 1
fi

# Add step to the output file name if step is greater than 0
if [ "$step" -gt 0 ]; then
  output_file="${output_file}_step=${step}.json"
elif [[ "$output_file" != *.json ]]; then
  output_file="${output_file}.json"
fi

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
    --target_param_name $target_param_name \
    --distribution $distribution \
    --rl_model_path $rl_model_path \
    --checkpoint_dir $checkpoint_dir \
    --step $step \
    --selector_matrix_dir $selector_matrix_dir &>"$log_dir/generate_${timestamp}.log"
fi

# Generate paraphrases if PARAPHRASE is set to 1
if [ "$PARAPHRASE" -eq 1 ]; then

  # python scripts/paraphrase.py \
  #   --output_file $output_file \
  #   --lex 60 --order 0 &>"$log_dir/paraphrase_l60_${timestamp}.log"

  # python scripts/paraphrase.py \
  #   --output_file $output_file \
  #   --lex 20 --order 0 &>"$log_dir/paraphrase_l20_${timestamp}.log"

  python scripts/paraphrase_llm.py \
    --output_file $output_file --num_beams 3 &>"$log_dir/paraphrase_llm_${timestamp}.log"
fi

python -m scripts.test_watermarking_v1 \
  --output_file "$output_file" \
  --log_dir "$log_dir" &>"$log_dir/tw_${timestamp}.log"

# Evaluate perplexity if eval_ppl is set to 1
if [ "$eval_ppl" -eq 1 ]; then
  python scripts/evaluate_ppl.py \
    --batch_size 16 \
    --output_file $output_file
fi
