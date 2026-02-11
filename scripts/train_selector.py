#!/usr/bin/env python
"""Train a selector matrix from hidden states saved on disk."""

from __future__ import annotations
from src.mbmark import MbMark  # local import to avoid cost if not needed

import argparse
import json
import sys
from pathlib import Path

# Use a non-interactive backend so scripts can run on headless machines.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from tqdm import tqdm  # noqa: E402


PLOT_DIR_NAME = "plots"
RIDGE_ALPHA = 1e-3
HIDDEN_STATES_FILENAME = "hidden_states.pt"
PREFIXES_FILENAME = "prefixes.pt"
PROJECTION_EVAL_SAMPLES = 5_000
RESCALE_MATRIX = True
MIN_CLUSTER_SIZE = 10
MAX_PER_CLASS = 30_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a selector matrix from precomputed hidden states."
    )
    parser.add_argument("--root-dir", default=".",
                        help="Project root for data + outputs.")
    parser.add_argument("--dataset-name", default="Skylion007/openwebtext")
    parser.add_argument(
        "--model-name", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--num-samples", type=int, default=1_500_000)
    parser.add_argument("--k", type=int, default=256,
                        help="Number of selector buckets.")
    parser.add_argument(
        "--labeling",
        choices=["kmeans", "rand_proj"],
        default="kmeans",
        help="How to obtain pseudo-labels for hidden states.",
    )
    parser.add_argument("--prf-key", type=int, default=42,
                        help="Seed for label generation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sem-align",
        action="store_true",
        default=False,
        help="Use semantic alignment: project hidden states before clustering and regression.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-mpnet-base-v2",
        help="Embedding model name for semantic alignment.",
    )
    return parser.parse_args()


def load_hidden_states(path: Path, num_samples: int) -> torch.Tensor:
    hidden_states = torch.load(path)
    hidden_states = hidden_states.float()
    if num_samples:
        hidden_states = hidden_states[:num_samples]
    return hidden_states


def load_alignment_matrix(data_dir: Path, embedding_model: str) -> torch.Tensor | None:
    """Load the semantic alignment matrix if it exists."""
    align_path = data_dir / f"align_ridge_{embedding_model}.pt"
    if not align_path.exists():
        raise FileNotFoundError(
            f"Alignment matrix not found at {align_path}. "
            f"Please run generate_alignment_ridge.py first."
        )
    print(f"Loading alignment matrix from {align_path}")
    W_align = torch.load(align_path)
    return W_align.float()


def generate_rand_proj_labels(hidden_states: torch.Tensor, prf_key: int, k: int) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(prf_key)
    W_random, _ = torch.linalg.qr(torch.randn(
        hidden_states.shape[1], k, generator=rng))
    logits = F.linear(hidden_states, W_random.T)
    return torch.argmax(logits, dim=1)


def generate_kmeans_labels(
    hidden_states: torch.Tensor, prf_key: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=prf_key,
        init="k-means++",
        max_no_improvement=20,
        batch_size=8192 * 10,
        max_iter=100,
        verbose=1,
    )
    kmeans.fit(hidden_states.cpu().numpy())
    labels = torch.tensor(
        kmeans.labels_, device=hidden_states.device, dtype=torch.long)
    centroids = torch.tensor(kmeans.cluster_centers_,
                             device=hidden_states.device, dtype=torch.float32)
    return labels, centroids


def clean_clusters(
    all_labels: torch.Tensor, all_hidden_states: torch.Tensor, min_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove tiny clusters and reindex labels to be contiguous."""
    device = all_labels.device
    max_label = all_labels.max().item()
    label_counts = torch.bincount(all_labels, minlength=max_label + 1)

    valid_mask = label_counts >= min_size
    valid_labels = torch.nonzero(valid_mask).squeeze(1)

    mapping = -torch.ones(max_label + 1, dtype=torch.long, device=device)
    mapping[valid_labels] = torch.arange(valid_labels.size(0), device=device)

    remapped_labels = mapping[all_labels]
    valid_idx = remapped_labels != -1

    return remapped_labels[valid_idx], all_hidden_states[valid_idx], valid_idx


def balanced_downsample(
    hidden: torch.Tensor, labels: torch.Tensor, max_per_class: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap each class to max_per_class points while keeping class balance."""
    unique_classes, counts = torch.unique(labels, return_counts=True)
    order = torch.argsort(labels)  # stable ordering of each class block

    selected_idx = []
    start = 0
    for count in counts.tolist():
        end = start + count
        class_block = order[start:end]
        take = min(max_per_class, count)
        if take < count:
            perm = torch.randperm(count, device=labels.device)[:take]
            class_block = class_block[perm]
        selected_idx.append(class_block)
        start = end

    selected_idx = torch.cat(selected_idx)
    return hidden[selected_idx], labels[selected_idx], selected_idx


def ridge_regression(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> torch.Tensor:
    d_in = X.shape[1]
    I = torch.eye(d_in, device=X.device)
    return torch.linalg.solve(X.T @ X + alpha * I, X.T @ Y)


def rescale(final_matrix: torch.Tensor, k: int, X_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
    X_projected = F.linear(X_train, final_matrix)
    max_values, _ = torch.max(X_projected, dim=1)
    cluster_max_means = []
    for cls in range(k):
        values = max_values[y_train == cls]
        if values.numel() == 0:
            raise ValueError(f"Cluster {cls} is empty after downsampling.")
        cluster_max_means.append(values.mean())

    cluster_max_means = torch.tensor(
        cluster_max_means, device=final_matrix.device).pow(-1)
    scale = torch.diag(cluster_max_means)
    return scale @ final_matrix


def compute_accuracy(final_matrix: torch.Tensor, X: torch.Tensor, y_true: torch.Tensor) -> float:
    X_projected = F.linear(X, final_matrix)
    y_pred = torch.argmax(X_projected, dim=1)
    return (y_pred == y_true).float().mean().item()


def measure_one_hotness(
    logits: torch.Tensor, labels: torch.Tensor | None = None, p: int = 1
) -> dict[str, float | None]:
    if logits.ndim < 2:
        raise ValueError("logits must have shape [..., K]")

    N = logits.shape[:-1].numel()
    K = logits.shape[-1]
    logits2d = logits.reshape(N, K)

    top2 = logits2d.topk(2, dim=1).values
    winner = top2[:, 0]
    runner_up = top2[:, 1]
    mean_gap = (winner - runner_up).mean().item()
    sharpness = (winner / (logits2d.sum(dim=1) - winner + 1e-8)).mean().item()

    results: dict[str, float | None] = {
        f"l{p}_error": None,
        "top1_accuracy": None,
        "mean_gap": mean_gap,
        "sharpness": sharpness,
    }

    if labels is not None:

        if labels.numel() != N:
            raise ValueError(
                "labels must have the same number of elements as logits rows")
        if labels.ndim != 1:
            labels = labels.view(-1)
        one_hot = F.one_hot(labels.to(torch.long),
                            num_classes=K).to(dtype=logits.dtype)
        diff = logits2d - one_hot
        if p == 1:
            err = diff.abs().mean().item()
        elif p == 2:
            err = diff.pow(2).mean().sqrt().item()
        else:
            err = diff.abs().pow(p).mean().pow(1.0 / p).item()
        results[f"l{p}_error"] = err
        results["top1_accuracy"] = (logits2d.argmax(
            dim=1) == labels).float().mean().item()

    return results


def save_pie_chart(counts: np.ndarray, labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct="%1.1f%%")
    plt.title("Cluster Distribution")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_histogram(values: torch.Tensor, path: Path, bins: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(values.cpu().numpy().flatten(), density=True,
             bins=bins, alpha=0.7, color="blue")
    plt.title("Projections Histogram")
    plt.xlabel("Projection Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def train(train_hidden_states, train_labels, k, W_align=None):
    # Downsample to cap per-class size for efficiency.
    train_hidden_states_ds, train_labels_ds, _ = balanced_downsample(
        train_hidden_states, train_labels, max_per_class=MAX_PER_CLASS
    )

    print(
        f"Train size: {len(train_hidden_states)} ({len(train_hidden_states_ds)} after downsample), "
    )
    # Train ridge regression selector.
    train_labels_one_hot = F.one_hot(train_labels_ds, num_classes=k).float()
    W_selector = ridge_regression(
        train_hidden_states_ds,
        train_labels_one_hot,
        alpha=RIDGE_ALPHA,
    )
    if torch.isnan(W_selector).any():
        raise ValueError("NaNs found in regression weights.")

    if RESCALE_MATRIX:
        final_matrix = rescale(
            W_selector.T, k, train_hidden_states_ds, train_labels_ds)
    else:
        final_matrix = W_selector.T

    if W_align is not None:
        final_matrix = final_matrix @ W_align.T

    if torch.isnan(final_matrix).any():
        raise ValueError("NaNs found after post processing.")
    return final_matrix


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_suffix = args.dataset_name.split("/")[-1]
    model_suffix = args.model_name.split("/")[-1]
    data_dir = Path(args.root_dir) / "data" / \
        f"{dataset_suffix}_{model_suffix}"
    model_dir = Path(args.root_dir) / "saved_models" / \
        f"{dataset_suffix}_{model_suffix}"
    hidden_states_path = data_dir / HIDDEN_STATES_FILENAME
    plot_dir = Path(args.root_dir) / PLOT_DIR_NAME / \
        f"{dataset_suffix}_{model_suffix}" / f"k_{args.k}"

    hidden_states = load_hidden_states(hidden_states_path, args.num_samples)
    print(f"Loaded hidden states: {hidden_states.shape}")

    # Load alignment matrix if semantic alignment is enabled
    W_align = None
    hidden_states_for_clustering = hidden_states.clone()
    if args.sem_align:
        W_align = load_alignment_matrix(data_dir, args.embedding_model)
        print(f"Alignment matrix shape: {W_align.shape}")
        # Project hidden states to embedding space for clustering
        hidden_states_for_clustering = F.linear(hidden_states, W_align.T)
        print(
            f"Projected hidden states shape: {hidden_states_for_clustering.shape}")

    l2_norm = hidden_states_for_clustering.norm(p=2, dim=1)
    norm_mean = l2_norm.mean().item()
    norm_std = l2_norm.std().item()
    print(f"L2 Norm Mean: {norm_mean:.4f}, L2 Norm Std: {norm_std:.4f}")

    prf_key = args.prf_key % 2**64
    if args.labeling == "kmeans":
        if args.sem_align:
            hidden_states_for_clustering = F.normalize(
                hidden_states_for_clustering, dim=1)

        all_labels, _ = generate_kmeans_labels(
            hidden_states_for_clustering, prf_key, args.k)
        all_labels_cleaned, hidden_states_for_clustering_cleaned, valid_idx = clean_clusters(
            all_labels, hidden_states_for_clustering, min_size=MIN_CLUSTER_SIZE)
        # Sync original hidden states with cleaned labels
        hidden_states = hidden_states[valid_idx]
        all_labels = all_labels_cleaned
        hidden_states_for_clustering = hidden_states_for_clustering_cleaned
        k = int(all_labels.max().item() + 1)
        print(f"Reduced number of classes after cleaning: {k}")
    elif args.labeling == "rand_proj":
        all_labels = generate_rand_proj_labels(
            hidden_states_for_clustering, prf_key, args.k)
        k = args.k

    unique, counts = torch.unique(all_labels, return_counts=True)
    counts_np = counts.cpu().numpy()
    semalign_prefix = f"semalign_{args.embedding_model}_" if args.sem_align else ""
    save_pie_chart(counts_np, unique.cpu().numpy(),
                   plot_dir / f"{semalign_prefix}cluster_distribution_{args.labeling}_k{args.k}.png")
    print(f"Cluster counts: {counts_np.tolist()}")

    train_idx, val_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=0.25,
        stratify=all_labels.cpu().numpy(),
        random_state=args.seed,
    )
    train_hidden_states = hidden_states_for_clustering[train_idx]
    train_hidden_states_orig = hidden_states[train_idx]
    train_labels = all_labels[train_idx]
    val_hidden_states = hidden_states[val_idx]
    val_labels = all_labels[val_idx]

    final_matrix = train(train_hidden_states, train_labels, k,
                         W_align=W_align)

    train_acc = compute_accuracy(
        final_matrix, train_hidden_states_orig, train_labels)
    val_acc = compute_accuracy(final_matrix, val_hidden_states, val_labels)

    logits_val = F.linear(val_hidden_states, final_matrix)
    one_hotness_metrics = measure_one_hotness(
        logits_val, labels=val_labels, p=1)

    watermark_matrix = MbMark._make_watermarking_matrix(
        32000, 1, 0.25, seed=prf_key, n_clusters=k
    )
    delta_w = watermark_matrix @ final_matrix
    projections = F.linear(
        val_hidden_states[:PROJECTION_EVAL_SAMPLES], delta_w)
    save_histogram(projections, plot_dir /
                   f"{semalign_prefix}projections_hist.png")

    extreme_threshold = 3.0
    extreme_counts = (projections.abs() > extreme_threshold).sum(dim=1)
    extreme_fraction = extreme_counts.float().mean().item() / projections.size(1)

    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_semalign_{args.embedding_model}" if args.sem_align else ""
    selector_path = model_dir / f"selector_matrix_k{k}{suffix}.pth"
    torch.save(final_matrix, selector_path)
    print(f"Saved selector matrix to {selector_path}")
    if args.sem_align:
        print(f"Note: This matrix is composed (selector @ alignment) and maps directly from hidden states to labels.")

    metrics = {
        "k": k,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "one_hotness": one_hotness_metrics,
        "projection_extreme_fraction": extreme_fraction,
        "projection_threshold": extreme_threshold,
        "sem_align": args.sem_align,
        "embedding_model": args.embedding_model if args.sem_align else None,
    }

    metrics_path = model_dir / f"selector_metrics_k{k}{suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
