import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from rich import print as rprint
from typing import Dict, List
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="Model name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str,
                        help="Dataset name", default="Skylion007/openwebtext")

    parser.add_argument("--num_samples", type=int,
                        help="Number of samples", default=1000)
    args = parser.parse_args()
    
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset,
                           split="train", streaming=True, trust_remote_code=True)

    dataset = dataset.filter(lambda x: len(
        tokenizer(x["text"])['input_ids']) >= 256)

    dataset = dataset.shuffle(seed=42)

    count_dataloader = torch.utils.data.DataLoader(dataset, batch_size=512)
    token_count = torch.zeros(tokenizer.vocab_size)
    pbar = tqdm(total=args.num_samples)

    for batch in count_dataloader:
        inputs = tokenizer(batch["text"], truncation=True,
                           max_length=256, return_tensors="pt")
        tokens = inputs["input_ids"].flatten()
        token_count.scatter_add_(
            0, tokens, torch.ones_like(tokens, dtype=torch.float32))
        pbar.update(1)
        if pbar.n >= args.num_samples:
            break

    pbar.close()

    dataset_suffix = args.dataset.split("/")[-1]
    model_suffix = args.model.split("/")[-1]
    torch.save(
        token_count, f"saved_models/{dataset_suffix}_{model_suffix}_token_freq.pt")
