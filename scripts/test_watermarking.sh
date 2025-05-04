#!/bin/bash

# Default parameter values
DELTA=3.0
GAMMA=0.5
NUM_SAMPLES=500
PARAPHRASE=0
SEED=15485863
watermark="mb"
generate=1
align=0
train=0
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
  --seed)
    SEED="$2"
    shift
    ;;
  --align)
    align="$2"
    shift
    ;;
  --train)
    train="$2"
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
if [ -z "$output_dir" ]; then
  output_dir="output"
fi

log_dir="${output_dir}/logs"
mkdir -p "$log_dir"


timestamp=$(date +"%Y%m%d_%H%M%S")

output_file="${output_dir}/output_align=${align}_delta=${DELTA}_gamma=${GAMMA}_k=${K}_seed=${SEED}_watermark=${watermark}.json"

if [ "$train" -eq 1 ]; then
 PRF_KEY=$SEED ALIGN=$align OUTPUT_FILE=$output_file K=$K papermill notebooks/mse_v1.ipynb "$log_dir/mse_$timestamp.ipynb"
fi

if [ "$generate" -eq 1 ]; then
  python -m scripts.generate_samples --num_samples $NUM_SAMPLES \
  --output_file $output_file \
  --delta $DELTA \
  --gamma $GAMMA \
  --hash_key $SEED \
  --watermark $watermark \
 
  OUTPUT_FILE=$output_file papermill notebooks/compute_base_statistics.ipynb "$log_dir/bs_$timestamp.ipynb"

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

OUTPUT_FILE=$output_file papermill notebooks/test_watermarking_v1.ipynb "$log_dir/tw_$timestamp.ipynb"

python scripts/evaluate_ppl.py \
  --batch_size 16 \
  --output_file $output_file