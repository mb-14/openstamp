"""
Generate embeddings with prefix-reuse optimization.

Supported embedding models:
  - Qwen/Qwen3-Embedding-{0.6B,4B,8B}  (last-token pool via forward pass)
  - jinaai/jina-embeddings-v5-text-small (model.encode API)
  - intfloat/multilingual-e5-large-instruct (mean pool via forward pass)
  - BAAI/bge-m3 (CLS pool via forward pass; XLM-RoBERTa backbone)

Loads prefixes.pt (LLM token IDs), decodes to text, tokenizes with the
embedding model tokenizer, builds a prefix tree, and runs inference only on
maximal prefixes to save computation.
"""
import os
import random
import argparse
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from rich import print as rprint
from transformers import AutoTokenizer, AutoModel


# Known presets; other HuggingFace IDs fall back to family heuristics.
MODEL_PRESETS = {
    "Qwen/Qwen3-Embedding-0.6B": {"family": "qwen3", "max_length": 8192},
    "Qwen/Qwen3-Embedding-4B": {"family": "qwen3", "max_length": 8192},
    "Qwen/Qwen3-Embedding-8B": {"family": "qwen3", "max_length": 8192},
    "jinaai/jina-embeddings-v5-text-small": {
        "family": "jina_v5",
        "max_length": 32768,
        "default_task": "text-matching",
    },
    "intfloat/multilingual-e5-large-instruct": {
        "family": "e5",
        "max_length": 512,
        "pooling": "mean",
    },
    "BAAI/bge-m3": {
        "family": "bge",
        "max_length": 8192,
        "pooling": "cls",
    },
}

JINA_V5_TASKS = ("retrieval", "text-matching", "classification", "clustering")


def detect_model_family(model_name: str) -> str:
    preset = MODEL_PRESETS.get(model_name)
    if preset:
        return preset["family"]
    lower = model_name.lower()
    if "jina-embeddings-v5" in lower or "jina_embeddings_v5" in lower:
        return "jina_v5"
    if "multilingual-e5" in lower or "e5-large" in lower or "e5-base" in lower:
        return "e5"
    if "bge-m3" in lower or "bge_m3" in lower:
        return "bge"
    if "/bge-" in lower or lower.startswith("bge-"):
        return "bge"
    if "qwen3-embedding" in lower or "qwen3_embedding" in lower:
        return "qwen3"
    return "qwen3"


def pooling_for_family(family: str, preset: dict) -> str:
    if preset.get("pooling"):
        return preset["pooling"]
    if family == "e5":
        return "mean"
    if family == "bge":
        return "cls"
    return "last_token"


def resolve_embed_dim(model, family: str, truncate_dim: Optional[int]) -> int:
    if truncate_dim is not None:
        return truncate_dim
    if family == "jina_v5":
        for attr in ("embedding_dim", "hidden_size"):
            if hasattr(model.config, attr):
                return getattr(model.config, attr)
        return 1024
    return model.config.hidden_size


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool last non-pad token; supports left- or right-padded batches."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def mean_pool_sequence(hidden_states: torch.Tensor, length: int) -> torch.Tensor:
    """Mean-pool over the first `length` tokens (E5-style)."""
    if length <= 0:
        raise ValueError("length must be positive for mean pooling")
    return hidden_states[:length].mean(dim=0)


def pool_sequence(hidden_states: torch.Tensor, length: int, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        return mean_pool_sequence(hidden_states, length)
    if pooling == "cls":
        return hidden_states[0]
    return hidden_states[length - 1]


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
    pooling: str = "last_token",
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
        last_hidden = outputs.last_hidden_state

        for j, prefix in enumerate(batch):
            prefix_tuple = tuple(prefix)
            for orig_idx, true_len in source_prefix_to_indices[prefix_tuple]:
                embeddings[orig_idx] = pool_sequence(
                    last_hidden[j], true_len, pooling
                ).float().cpu()

        pbar.update(len(batch))

    pbar.close()
    return embeddings


@torch.no_grad()
def compute_embeddings_jina(
    model,
    tokenizer,
    all_input_ids,
    longest_prefixes,
    prefix_to_indices,
    prefix_to_source,
    embed_dim,
    batch_size,
    task,
    truncate_dim=None,
):
    embeddings = torch.zeros(all_input_ids.size(0), embed_dim, dtype=torch.float32)

    source_prefix_to_indices = defaultdict(list)
    for sub_prefix in prefix_to_source:
        source_prefix = prefix_to_source[sub_prefix]
        source_prefix_to_indices[source_prefix].extend(prefix_to_indices[sub_prefix])

    pbar = tqdm(total=len(longest_prefixes), desc="Computing embeddings (Jina)")

    for i in range(0, len(longest_prefixes), batch_size):
        batch = longest_prefixes[i : i + batch_size]
        texts = [
            tokenizer.decode(list(prefix), skip_special_tokens=True).strip() or " "
            for prefix in batch
        ]
        encode_kwargs = {"texts": texts, "task": task}
        if truncate_dim is not None:
            encode_kwargs["truncate_dim"] = truncate_dim
        batch_emb = model.encode(**encode_kwargs)
        if not isinstance(batch_emb, torch.Tensor):
            batch_emb = torch.tensor(batch_emb, dtype=torch.float32)
        else:
            batch_emb = batch_emb.float().cpu()

        for j, prefix in enumerate(batch):
            prefix_tuple = tuple(prefix)
            for orig_idx, _true_len in source_prefix_to_indices[prefix_tuple]:
                embeddings[orig_idx] = batch_emb[j]

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
    pooling: str = "last_token",
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
        expected = pool_sequence(outputs.last_hidden_state[0], length, pooling).float().cpu()
        expected = F.normalize(expected.unsqueeze(0), p=2, dim=1).squeeze(0)
        actual = embeddings[idx]
        if not torch.allclose(expected, actual, atol=1e-2):
            failed.append(idx)

    if failed:
        rprint(f"[bold red]Validation failed for {len(failed)} indices.[/bold red]")
    else:
        rprint("[bold green]All sampled prefixes passed validation.[/bold green]")


@torch.no_grad()
def validate_embeddings_jina(
    model,
    tokenizer,
    all_input_ids,
    pad_token_id,
    embeddings,
    task,
    truncate_dim=None,
    num_samples=10,
):
    rprint("[bold yellow]Running Jina validation on sampled prefixes...[/bold yellow]")
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
        text = tokenizer.decode(
            all_input_ids[idx, :length].tolist(), skip_special_tokens=True
        ).strip() or " "
        encode_kwargs = {"texts": [text], "task": task}
        if truncate_dim is not None:
            encode_kwargs["truncate_dim"] = truncate_dim
        expected = model.encode(**encode_kwargs)
        if not isinstance(expected, torch.Tensor):
            expected = torch.tensor(expected, dtype=torch.float32)
        else:
            expected = expected.float().cpu()
        expected = F.normalize(expected[0], p=2, dim=0)
        actual = embeddings[idx]
        if not torch.allclose(expected, actual, atol=1e-2):
            failed.append(idx)

    if failed:
        rprint(f"[bold red]Validation failed for {len(failed)} indices.[/bold red]")
    else:
        rprint("[bold green]All sampled prefixes passed validation.[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings with prefix reuse (Qwen3 / Jina v5 / E5 / BGE)."
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
        help=(
            "HuggingFace embedding model. Supported presets: "
            "Qwen/Qwen3-Embedding-{0.6B,4B,8B}, "
            "jinaai/jina-embeddings-v5-text-small, "
            "intfloat/multilingual-e5-large-instruct, "
            "BAAI/bge-m3"
        ),
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
        default=None,
        help="Max sequence length (default: model preset or 8192)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=JINA_V5_TASKS,
        help="Jina v5 task (default: text-matching)",
    )
    parser.add_argument(
        "--truncate_dim",
        type=int,
        default=None,
        help="Matryoshka output dimension (Jina v5 / Qwen3 MRL)",
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

    preset = MODEL_PRESETS.get(args.embedding_model, {})
    family = detect_model_family(args.embedding_model)
    max_length = args.max_length or preset.get("max_length", 8192)
    pooling = pooling_for_family(family, preset)
    jina_task = args.task or preset.get("default_task", "text-matching")

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rprint(f"[bold green]Embedding model:[/bold green] {args.embedding_model}")
    rprint(f"[bold green]Model family:[/bold green] {family}")
    if family != "jina_v5":
        rprint(f"[bold green]Pooling:[/bold green] {pooling}")

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

    rprint("[bold green]Loading embedding tokenizer and model...[/bold green]")
    needs_remote_code = family in ("qwen3", "jina_v5")
    emb_tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model,
        padding_side="right",
        trust_remote_code=needs_remote_code,
    )
    if emb_tokenizer.pad_token is None:
        emb_tokenizer.pad_token = emb_tokenizer.eos_token

    model_kwargs = {}
    if needs_remote_code:
        model_kwargs["trust_remote_code"] = True
    if family == "jina_v5":
        model_kwargs["dtype"] = torch.bfloat16
        if args.flash_attention_2:
            model_kwargs["_attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        if args.flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModel.from_pretrained(args.embedding_model, **model_kwargs)
    model = model.to(device)
    model.eval()

    embed_dim = resolve_embed_dim(model, family, args.truncate_dim)

    rprint(f"[bold cyan]Tokenizing (max_length={max_length})...[/bold cyan]")
    all_input_ids = tokenize_in_batches(
        tokenizer=emb_tokenizer,
        texts=decoded_texts,
        max_length=max_length,
        batch_size=args.batch_size,
    )
    pad_token_id = emb_tokenizer.pad_token_id

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
    if family == "jina_v5":
        rprint(f"[bold cyan]Jina task: {jina_task}[/bold cyan]")
        embeddings = compute_embeddings_jina(
            model=model,
            tokenizer=emb_tokenizer,
            all_input_ids=all_input_ids,
            longest_prefixes=longest_prefixes,
            prefix_to_indices=prefix_to_indices,
            prefix_to_source=prefix_to_source,
            embed_dim=embed_dim,
            batch_size=args.batch_size,
            task=jina_task,
            truncate_dim=args.truncate_dim,
        )
    else:
        embeddings = compute_embeddings(
            model=model,
            tokenizer=emb_tokenizer,
            all_input_ids=all_input_ids,
            longest_prefixes=longest_prefixes,
            prefix_to_indices=prefix_to_indices,
            prefix_to_source=prefix_to_source,
            embed_dim=embed_dim,
            batch_size=args.batch_size,
            device=device,
            pooling=pooling,
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
        if family == "jina_v5":
            validate_embeddings_jina(
                model=model,
                tokenizer=emb_tokenizer,
                all_input_ids=all_input_ids,
                pad_token_id=pad_token_id,
                embeddings=embeddings,
                task=jina_task,
                truncate_dim=args.truncate_dim,
                num_samples=min(args.batch_size, total_indices),
            )
        else:
            validate_embeddings(
                model=model,
                tokenizer=emb_tokenizer,
                all_input_ids=all_input_ids,
                pad_token_id=pad_token_id,
                embeddings=embeddings,
                num_samples=min(args.batch_size, total_indices),
                device=device,
                pooling=pooling,
            )

    model_name = args.embedding_model.split("/")[-1]
    out_path = os.path.join(args.dataset_path, f"embeddings_{model_name}.pt")
    torch.save(embeddings, out_path)
    rprint(f"[bold green]Saved embeddings to {out_path}[/bold green]")


if __name__ == "__main__":
    main()
