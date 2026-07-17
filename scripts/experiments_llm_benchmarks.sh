#!/bin/bash
# Thin wrapper around lm_eval for arbitrary HF model paths.
# Prefer scripts/run_downstream_evals.sh for the Unremovable pipeline.
#
# Usage:
#   models=("meta-llama/Llama-2-7b-hf" "path/to/watermarked") \
#     ./scripts/experiments_llm_benchmarks.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${models+x}" ]]; then
  models=(
    "meta-llama/Llama-2-7b-hf"
    "mistralai/Mistral-7B-v0.3"
  )
fi

datasets=(boolq arc_challenge hellaswag)
num_fewshot="${NUM_FEWSHOT:-5}"
batch_size="${BATCH_SIZE:-8}"
results_root="${RESULTS_ROOT:-results}"

if ! command -v lm_eval >/dev/null 2>&1; then
  echo "lm_eval not found. Install with: pip install lm-eval" >&2
  exit 1
fi

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "Running model=$model on benchmark=$dataset"
    lm_eval --model hf --model_args "pretrained=${model},dtype=bfloat16" --tasks "$dataset" \
      --num_fewshot "$num_fewshot" \
      --batch_size "$batch_size" \
      --output_path "${results_root}/${dataset}" --log_samples
  done
done
