#!/usr/bin/env python
"""
Convert a streaming text dataset into a balanced tensor of token prefixes.

This mirrors the logic from `notebooks/preprocess.ipynb` so it can be run as a
stand-alone script.
"""

import argparse
import os
from typing import List

import einops
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List, Union
torch.manual_seed(42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate balanced token prefixes.")
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL", "meta-llama/Llama-2-7b-hf"),
        help="Tokenizer checkpoint to use.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Skylion007/openwebtext",
        help="HF dataset name to stream from.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=int(os.getenv("TOTAL", 1000)),
        help="Number of batches to take from the dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for streaming from the dataset.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum token length per sample.",
    )
    parser.add_argument(
        "--balance-interval",
        type=int,
        default=25,
        help="How often (in batches) to balance the collected prefixes.",
    )
    parser.add_argument(
        "--output-root",
        default="data",
        help="Directory where the prefixes tensor will be stored.",
    )
    return parser.parse_args()


def prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def filter_length(example, tokenizer, max_seq_len: int) -> bool:
    return len(tokenizer(example["text"])["input_ids"]) >= max_seq_len


def to_right_padding(input_ids: torch.Tensor, pad_token_id: int, padding_side: str) -> torch.Tensor:
    """Ensure sequences are right padded so downstream slicing logic stays valid."""
    if padding_side != "left":
        return input_ids

    seq_lens = (input_ids != pad_token_id).sum(dim=1)
    max_len = input_ids.size(1)
    right_padded = torch.full_like(input_ids, pad_token_id)

    for row_idx, seq_len in enumerate(seq_lens.tolist()):
        if seq_len == 0:
            continue
        right_padded[row_idx, :seq_len] = input_ids[row_idx, max_len - seq_len : max_len]

    return right_padded



def balance_data(
    prefix_batches: Union[List[torch.Tensor], torch.Tensor],
    pad_token_id: int,
) -> List[torch.Tensor]:
    """
    Make the class distribution (based on the last non-pad token in each row)
    more balanced, while being more CPU-efficient than the original version.

    - Works with either a list of tensors or a single tensor.
    - Avoids flip+argmax and per-class boolean masks.
    """

    # Allow both list[Tensor] and Tensor as input
    if isinstance(prefix_batches, list):
        prefixes_vec = torch.cat(prefix_batches, dim=0)
    else:
        prefixes_vec = prefix_batches

    # prefixes_vec: [N, L]
    non_pad_mask = prefixes_vec != pad_token_id       # [N, L] bool
    lengths = non_pad_mask.sum(dim=1)                 # [N], # of non-pad tokens

    # last index = length - 1, but clamp for empty rows
    # lengths == 0 -> last index 0 to avoid indexing error
    last_token_indices = torch.clamp(lengths, min=1) - 1    # [N]
    token_ids_vec = prefixes_vec[torch.arange(prefixes_vec.size(0)), last_token_indices]

    sample_count = prefixes_vec.size(0)

    # Group by class using sort + unique_consecutive to avoid many masks
    sorted_tokens, sorted_indices = torch.sort(token_ids_vec)  # both [N]
    unique_classes, class_counts = torch.unique_consecutive(
        sorted_tokens,
        return_counts=True
    )
    num_classes = unique_classes.size(0)
    samples_per_class = max(1, sample_count // max(1, num_classes))

    # Iterate over contiguous segments instead of (token_ids_vec == cls) masks
    selected_indices_per_class = []
    cum_counts = class_counts.cumsum(dim=0)

    start = 0
    for count in class_counts:
        end = start + count
        class_indices = sorted_indices[start:end]  # indices for this class

        if count > samples_per_class:
            # Randomly choose a subset for this class
            perm = torch.randperm(count)
            class_indices = class_indices[perm[:samples_per_class]]

        selected_indices_per_class.append(class_indices)
        start = end

    selected_indices = torch.cat(selected_indices_per_class, dim=0)

    # Shuffle globally; keep at most the original number of samples
    if selected_indices.numel() > 0:
        perm = torch.randperm(selected_indices.size(0))
        selected_indices = selected_indices[perm][:sample_count]

    balanced = prefixes_vec[selected_indices]
    # Keep your original return type: list[Tensor]
    return [balanced]

def collect_prefixes(args: argparse.Namespace, tokenizer) -> torch.Tensor:
    dataset = load_dataset(
        args.dataset_name,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    dataset = dataset.filter(lambda example: filter_length(example, tokenizer, args.max_seq_len))
    dataset = dataset.shuffle(seed=42).take(args.total * args.batch_size)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    progress_bar = tqdm(dataloader, total=args.total, desc="Collecting prefixes")

    all_prefixes: List[torch.Tensor] = []
    for batch_index, batch in enumerate(dataloader):
        text = batch["text"]
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt",
        )
        input_ids = to_right_padding(encoded["input_ids"], tokenizer.pad_token_id, tokenizer.padding_side)

        prefixes = [input_ids[:, : i + 1].T for i in range(input_ids.shape[1])]
        padded_prefixes = torch.nn.utils.rnn.pad_sequence(prefixes, padding_value=tokenizer.pad_token_id)
        padded_prefixes = einops.rearrange(padded_prefixes, "seq_len seqs batch -> (batch seqs) seq_len")
        padded_prefixes = torch.unique(padded_prefixes, dim=0)
        all_prefixes.append(padded_prefixes)

        if batch_index % args.balance_interval == 0 and all_prefixes:
            all_prefixes = balance_data(torch.cat(all_prefixes, dim=0), tokenizer.pad_token_id)
            print(f"Batch {batch_index} - Data balanced. Size: {all_prefixes[0].size(0)}")
        progress_bar.update()
    progress_bar.close()

    if not all_prefixes:
        raise RuntimeError("No prefixes were collected from the dataset.")

    return torch.cat(all_prefixes, dim=0)


def save_prefixes(prefixes: torch.Tensor, dataset_name: str, model_name: str, output_root: str) -> str:
    dataset_suffix = dataset_name.split("/")[-1]
    model_suffix = model_name.split("/")[-1]
    prefixes_path = os.path.join(output_root, f"{dataset_suffix}_{model_suffix}", "prefixes.pt")
    os.makedirs(os.path.dirname(prefixes_path), exist_ok=True)
    torch.save(prefixes, prefixes_path)
    return prefixes_path


def main():
    args = parse_args()
    tokenizer = prepare_tokenizer(args.model_name)
    prefixes = collect_prefixes(args, tokenizer)
    output_path = save_prefixes(prefixes, args.dataset_name, args.model_name, args.output_root)
    print(f"Saved {prefixes.size(0)} prefixes to {output_path}")


if __name__ == "__main__":
    main()
