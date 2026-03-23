"""
Generate embeddings using Qwen3-Embedding-8B with prefix reuse optimization.
Loads prefixes.pt (LLM token IDs), decodes to text, tokenizes with Qwen3,
builds a prefix tree in Qwen3 token space, and runs the model only on
maximal prefixes to save computation.
"""
import os
import random
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from rich import print as rprint
from transformers import AutoTokenizer, AutoModel


def load_decoder_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prefix_tree(all_input_ids, pad_token_id):
    pad_mask = (all_input_ids != pad_token_id)
    lengths = pad_mask.sum(dim=1)

    trie = {}
    prefix_to_indices = defaultdict(list)

    for idx in range(all_input_ids.size(0)):
        length = lengths[idx].item()
        if length == 0:
            continue
        seq = tuple(all_input_ids[idx, :length].tolist())
        prefix_to_indices[seq].append((idx, length))

        node = trie
        for token in seq:
            node = node.setdefault(token, {})
        node.setdefault("__end__", True)

    return trie, prefix_to_indices


def get_maximal_prefixes(trie):
    maximal = []

    def count_leaf_paths(node):
        children = [k for k in node if k != "__end__"]
        if not children:
            return 1
        return sum(count_leaf_paths(node[k]) for k in children)

    total = count_leaf_paths(trie)
    pbar = tqdm(total=total, desc="Finding maximal prefixes")

    def dfs(node, path):
        children = [k for k in node if k != "__end__"]
        if not children:
            maximal.append(tuple(path))
            pbar.update(1)
        else:
            for k in children:
                dfs(node[k], path + [k])

    dfs(trie, [])
    pbar.close()
    return maximal


def map_subprefixes_to_longest(prefix_to_indices, longest_prefixes):
    prefix_to_source = {}
    total = len(prefix_to_indices)
    with tqdm(total=total, desc="Mapping subprefixes") as pbar:
        for long_prefix in longest_prefixes:
            for sublen in range(1, len(long_prefix) + 1):
                sub = long_prefix[:sublen]
                if sub in prefix_to_indices and sub not in prefix_to_source:
                    prefix_to_source[sub] = long_prefix
                    pbar.update(1)
    return prefix_to_source


def tokenize_in_batches(tokenizer, texts, max_length, batch_size):
    all_token_ids = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing batches"):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
        )
        all_token_ids.extend(encoded["input_ids"])

    if not all_token_ids:
        return torch.empty((0, 0), dtype=torch.long)

    all_input_ids = pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in all_token_ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    return all_input_ids


@torch.no_grad()
def compute_embeddings(
    model,
    tokenizer,
    all_input_ids,
    longest_prefixes,
    prefix_to_indices,
    prefix_to_source,
    embed_dim,
    batch_size,
    device,
):
    embeddings = torch.zeros(
        all_input_ids.size(0), embed_dim, dtype=torch.float32, device="cpu"
    )

    source_prefix_to_indices = defaultdict(list)
    for sub_prefix in prefix_to_source:
        source_prefix = prefix_to_source[sub_prefix]
        source_prefix_to_indices[source_prefix].extend(
            prefix_to_indices[sub_prefix]
        )

    pbar = tqdm(total=len(longest_prefixes), desc="Computing embeddings")

    for i in range(0, len(longest_prefixes), batch_size):
        batch = longest_prefixes[i : i + batch_size]
        lengths = [len(p) for p in batch]
        max_len = max(lengths)

        input_ids = torch.full(
            (len(batch), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        for j, prefix in enumerate(batch):
            input_ids[j, : len(prefix)] = torch.tensor(
                prefix, dtype=torch.long, device=device
            )

        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        for j, prefix in enumerate(batch):
            prefix_tuple = tuple(prefix)
            for orig_idx, true_len in source_prefix_to_indices[prefix_tuple]:
                embeddings[orig_idx] = last_hidden[j, true_len - 1].float().cpu()

        pbar.update(len(batch))

    pbar.close()
    return embeddings


@torch.no_grad()
def validate_embeddings(
    model,
    tokenizer,
    all_input_ids,
    pad_token_id,
    embeddings,
    num_samples=10,
    device="cuda",
):
    rprint("[bold yellow]Running validation on sampled prefixes...[/bold yellow]")
    pad_mask = (all_input_ids != pad_token_id)
    lengths = pad_mask.sum(dim=1)
    indices = random.sample(
        range(len(all_input_ids)),
        k=min(num_samples, len(all_input_ids)),
    )

    failed = []
    for idx in indices:
        length = lengths[idx].item()
        if length == 0:
            continue
        input_ids = all_input_ids[idx, :length].unsqueeze(0).to(device)
        attention_mask = (input_ids != pad_token_id).long().to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        expected = outputs.last_hidden_state[0, length - 1].float().cpu()
        expected = F.normalize(expected.unsqueeze(0), p=2, dim=1).squeeze(0)
        actual = embeddings[idx]
        if not torch.allclose(expected, actual, atol=1e-2):
            failed.append(idx)

    if failed:
        rprint(f"[bold red]Validation failed for {len(failed)} indices.[/bold red]")
    else:
        rprint("[bold green]All sampled prefixes passed validation.[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings with Qwen3-Embedding-8B and prefix reuse."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset dir containing prefixes.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer used to decode the dataset (LLM tokenizer)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="Qwen3 embedding model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for embedding model forward passes",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 = all)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Max sequence length for Qwen3 tokenizer",
    )
    parser.add_argument(
        "--flash_attention_2",
        action="store_true",
        help="Use flash_attention_2 for acceleration",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation on a sample of embeddings",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rprint("[bold green]Loading decoder tokenizer...[/bold green]")
    decoder_tokenizer = load_decoder_tokenizer(args.tokenizer)

    rprint("[bold green]Loading dataset...[/bold green]")
    path = os.path.join(args.dataset_path, "prefixes.pt")
    all_llm_input_ids = torch.load(path, weights_only=True)
    if args.total_samples > 0:
        all_llm_input_ids = all_llm_input_ids[: args.total_samples]
    n_samples = all_llm_input_ids.size(0)

    rprint("[bold green]Decoding prefixes to text...[/bold green]")
    decoded_texts = decoder_tokenizer.batch_decode(
        all_llm_input_ids, skip_special_tokens=True
    )
    for i, text in enumerate(decoded_texts):
        if not text.strip():
            decoded_texts[i] = " "

    rprint("[bold green]Loading Qwen3 tokenizer and model...[/bold green]")
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model, padding_side="right"
    )
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if args.flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModel.from_pretrained(args.embedding_model, **model_kwargs)
    model = model.to(device)
    model.eval()

    embed_dim = model.config.hidden_size

    rprint("[bold cyan]Tokenizing with Qwen3 in batches...[/bold cyan]")
    all_input_ids = tokenize_in_batches(
        tokenizer=qwen_tokenizer,
        texts=decoded_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    pad_token_id = qwen_tokenizer.pad_token_id

    pad_mask = (all_input_ids != pad_token_id)
    lengths = pad_mask.sum(dim=1)
    if (lengths == 0).any():
        n_empty = (lengths == 0).sum().item()
        rprint(f"[yellow]Found {n_empty} empty sequences; skipping in tree.[/yellow]")

    rprint("[bold cyan]Building prefix tree...[/bold cyan]")
    trie, prefix_to_indices = build_prefix_tree(all_input_ids, pad_token_id)

    total_indices = sum(len(v) for v in prefix_to_indices.values())
    if total_indices < n_samples:
        rprint(
            f"[yellow]Only {total_indices} indices in tree (skipped empty); "
            f"filling {n_samples - total_indices} with zero embedding.[/yellow]"
        )

    rprint("[bold cyan]Identifying maximal prefixes for reuse...[/bold cyan]")
    longest_prefixes = get_maximal_prefixes(trie)
    prefix_to_source = map_subprefixes_to_longest(
        prefix_to_indices, longest_prefixes
    )
    assert len(prefix_to_source) == len(prefix_to_indices), (
        "Some subprefixes not mapped to longest prefixes."
    )

    rprint("[bold cyan]Computing embeddings...[/bold cyan]")
    embeddings = compute_embeddings(
        model=model,
        tokenizer=qwen_tokenizer,
        all_input_ids=all_input_ids,
        longest_prefixes=longest_prefixes,
        prefix_to_indices=prefix_to_indices,
        prefix_to_source=prefix_to_source,
        embed_dim=embed_dim,
        batch_size=args.batch_size,
        device=device,
    )

    rprint("[bold cyan]Normalizing embeddings...[/bold cyan]")
    valid_mask = (pad_mask.sum(dim=1) > 0)
    if valid_mask.any():
        if valid_mask.all():
            embeddings = F.normalize(embeddings, p=2, dim=1)
        else:
            norm_emb = F.normalize(embeddings[valid_mask], p=2, dim=1)
            embeddings[valid_mask] = norm_emb
            embeddings[~valid_mask] = 0.0
    else:
        embeddings.zero_()

    if args.validate and total_indices > 0:
        validate_embeddings(
            model=model,
            tokenizer=qwen_tokenizer,
            all_input_ids=all_input_ids,
            pad_token_id=pad_token_id,
            embeddings=embeddings,
            num_samples=min(args.batch_size, total_indices),
            device=device,
        )

    model_name = args.embedding_model.split("/")[-1]
    out_path = os.path.join(args.dataset_path, f"embeddings_{model_name}.pt")
    torch.save(embeddings, out_path)
    rprint(f"[bold green]Saved embeddings to {out_path}[/bold green]")


if __name__ == "__main__":
    main()
