import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich import print as rprint
import random
from collections import defaultdict


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    model.lm_head = torch.nn.Identity()
    return model, tokenizer


def build_prefix_tree(all_input_ids, pad_token_id):
    pad_mask = (all_input_ids != pad_token_id)
    lengths = pad_mask.sum(dim=1)

    trie = {}
    prefix_to_indices = defaultdict(list)

    for idx in range(all_input_ids.size(0)):
        length = lengths[idx].item()
        seq = tuple(all_input_ids[idx, :length].tolist())
        prefix_to_indices[seq].append((idx, length))

        node = trie
        for token in seq:
            node = node.setdefault(token, {})
        node.setdefault("__end__", True)

    return trie, prefix_to_indices


def get_maximal_prefixes(trie):
    maximal = []

    # Step 1: Count total number of leaf paths
    def count_leaf_paths(node):
        children = [k for k in node if k != "__end__"]
        if not children:
            return 1
        return sum(count_leaf_paths(node[k]) for k in children)

    total = count_leaf_paths(trie)

    # Step 2: DFS with tqdm
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
            for sublen in range(1, len(long_prefix)+1):
                sub = long_prefix[:sublen]
                if sub in prefix_to_indices and sub not in prefix_to_source:
                    prefix_to_source[sub] = long_prefix
                    pbar.update(1)
    return prefix_to_source


@torch.no_grad()
def compute_hidden_states(model, tokenizer, all_input_ids, longest_prefixes, prefix_to_indices, prefix_to_source, hidden_size, batch_size):
    index_to_hidden = torch.zeros(
        all_input_ids.size(0), hidden_size, dtype=torch.bfloat16)

    # Precompute: mapping from source_prefix → list of (orig_idx, true_len)
    source_prefix_to_indices = defaultdict(list)
    for sub_prefix in prefix_to_source:
        source_prefix = prefix_to_source[sub_prefix]
        source_prefix_to_indices[source_prefix].extend(
            prefix_to_indices[sub_prefix])

    pbar = tqdm(total=len(longest_prefixes), desc="Computing hidden states")

    for i in range(0, len(longest_prefixes), batch_size):
        batch = longest_prefixes[i:i+batch_size]
        lengths = torch.tensor([len(p) for p in batch])
        max_len = lengths.max().item()

        input_ids = torch.full((len(batch), max_len),
                               tokenizer.pad_token_id, dtype=torch.long)

        for j, prefix in enumerate(batch):
            input_ids[j, :len(prefix)] = torch.tensor(prefix, dtype=torch.long)

        input_ids = input_ids.cuda()

        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits  # (B, T, H)

        for j, prefix in enumerate(batch):
            prefix_tuple = tuple(prefix)
            for orig_idx, true_len in source_prefix_to_indices.get(prefix_tuple, []):
                index_to_hidden[orig_idx] = logits[j, true_len - 1].cpu()

        pbar.update(len(batch))

    pbar.close()
    return index_to_hidden


@torch.no_grad()
def validate_hidden_states(model, tokenizer, all_input_ids, hidden_states, prefix_to_source, num_samples=10):
    rprint("[bold yellow]Running validation on sampled prefixes...[/bold yellow]")
    pad_token_id = tokenizer.pad_token_id
    pad_mask = (all_input_ids != pad_token_id)
    lengths = pad_mask.sum(dim=1)
    indices = random.sample(range(len(all_input_ids)),
                            k=min(num_samples, len(all_input_ids)))

    for idx in indices:
        input_ids = all_input_ids[idx].unsqueeze(0).to("cuda")
        output = model(input_ids=input_ids)
        expected = output.logits[0, lengths[idx] -
                                 1].to("cpu", dtype=torch.bfloat16)
        actual = hidden_states[idx]
        assert torch.allclose(
            expected, actual, atol=1e-1), f"Validation failed for index {idx}."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--total_samples", type=int, default=-1,
                        help="Number of samples to process from the dataset")
    args = parser.parse_args()

    torch.manual_seed(42)

    rprint(f"[bold green]Loading model...[/bold green]")
    model, tokenizer = load_model(args.model)

    rprint(f"[bold green]Loading dataset...[/bold green]")
    path = os.path.join(args.dataset_path, "prefixes.pt")
    all_input_ids = torch.load(path)  # shape (N, 256)

    if args.total_samples > 0:
        all_input_ids = all_input_ids[:args.total_samples]

    rprint(f"[bold cyan]Building prefix tree...[/bold cyan]")
    trie, prefix_to_indices = build_prefix_tree(
        all_input_ids, tokenizer.pad_token_id)

    rprint(f"[bold cyan]Identifying longest prefixes for reuse...[/bold cyan]")
    longest_prefixes = get_maximal_prefixes(trie)
    prefix_to_source = map_subprefixes_to_longest(
        prefix_to_indices, longest_prefixes)

    rprint(f"[bold cyan]Computing hidden states...[/bold cyan]")
    hidden_states = compute_hidden_states(
        model=model,
        tokenizer=tokenizer,
        all_input_ids=all_input_ids,
        longest_prefixes=longest_prefixes,
        prefix_to_indices=prefix_to_indices,
        prefix_to_source=prefix_to_source,
        hidden_size=model.config.hidden_size,
        batch_size=args.batch_size,
    )

    if args.validate:
        validate_hidden_states(
            model, tokenizer, all_input_ids, hidden_states, prefix_to_source=prefix_to_source, num_samples=args.batch_size)

    model_name = args.model.split("/")[-1]
    out_path = os.path.join(
        args.dataset_path, f"hidden_states.pt")
    torch.save(hidden_states, out_path)

    rprint(f"[bold green]Saved hidden states to: {out_path}[/bold green]")
