# OpenStamp

A watermarking method for open-source Large Language Models.

## Setup Environment


```bash
conda create -n openstamp python=3.12 -y
conda activate openstamp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install -r requirements.txt

```

## Download models

By default the pipeline uses `meta-llama/Llama-2-13b-hf` as the PPL oracle and `Qwen/Qwen2.5-14B-Instruct` for paraphrasing. Download both from HuggingFace:

```bash
hf download meta-llama/Llama-2-13b-hf
hf download Qwen/Qwen2.5-14B-Instruct
```

Download testing models:

```bash
hf download meta-llama/Llama-2-7b-hf
hf download mistralai/Mistral-7B-v0.3
```

## Run experiments

Run the following command to generate watermarked samples and evaluate detection performance on the samples:

```bash
python scripts/run_config.py \
	--config experiment_configs/openstamp.yaml \
	--base_output_dir output/main \
	--num_samples 500 \
	--paraphrase \
	--eval_ppl
```

You can then aggregate metrics across seeds into a CSV file:

```bash
python scripts/aggregate_metrics.py \
	--input-dir output/main \
	--output-csv results/aggregated_metrics.csv
```

All available configs are in `experiment_configs/`:
- [OpenStamp](experiment_configs/openstamp.yaml)
- [KGW](experiment_configs/kgw.yaml)
- [KGW+LLR](experiment_configs/kgw_llr.yaml)
- [GaussMark](experiment_configs/gaussmark.yaml)
- [KGW Distilled](experiment_configs/distilled.yaml)
