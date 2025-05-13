#!/bin/bash

model="mistralai/Mistral-7B-v0.3"
# Get second part of the model name
dataset_path="data/openwebtext_Mistral-7B-v0.3"

#MODEL=$model papermill notebooks/preprocess.ipynb output/logs/pp.ipynb
python scripts/generate_hidden_states.py --dataset_path $dataset_path --batch_size 128 --model $model --total_samples 1500000
python scripts/generate_embeddings.py --dataset_path  $dataset_path --batch_size 512 --model sentence-transformers/all-mpnet-base-v2 --tokenizer $model --total_samples 1500000
MODEL=$model papermill notebooks/sem_align.ipynb output/logs/sa.ipynb