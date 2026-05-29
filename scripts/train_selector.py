#!/usr/bin/env python
"""Train a selector matrix from hidden states saved on disk."""

from __future__ import annotations
# local import to avoid cost if not needed
# from src.kmeans import MiniBatchKMeans
from src.kmeans_pytorch import MiniBatchKMeans
from src.openstamp import OpenStamp  # local import to avoid cost if not needed

import argparse
import json
import os
import sys
import time
from pathlib import Path


# Use a non-interactive backend so scripts can run on headless machines.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import safe_open  # noqa: E402

# from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from tqdm import trange  # noqa: E402


PLOT_DIR_NAME = "plots"
RIDGE_ALPHA = 1e-3
HIDDEN_STATES_FILENAME = "hidden_states.pt"
HIDDEN_STATES_SAFETENSORS_FILENAME = "hidden_states.safetensors"
PREFIXES_FILENAME = "prefixes.pt"
PROJECTION_EVAL_SAMPLES = 5_000
RESCALE_MATRIX = True
MIN_CLUSTER_SIZE = 10
MAX_PER_CLASS = 30_000
CLEAN_CHUNK_SIZE = 200_000
DEFAULT_PROJECTION_CHUNK_SIZE = 65_536


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
    parser.add_argument(
        "--hidden-states-format",
        choices=["auto", "pt", "safetensors"],
        default="auto",
        help="Hidden states format to load.",
    )
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
        "--align-method",
        choices=["ridge", "contrastive"],
        default="ridge",
        help="Alignment method to load when --sem-align is enabled.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-mpnet-base-v2",
        help="Embedding model name for semantic alignment.",
    )
    parser.add_argument(
        "--selector-dir",
        type=str,
        default=None,
        help="Directory to store selector_matrix and selector_metrics files.",
    )
    parser.add_argument(
        "--projection-chunk-size",
        type=int,
        default=DEFAULT_PROJECTION_CHUNK_SIZE,
        help="Rows per chunk when projecting hidden states with --sem-align.",
    )
    return parser.parse_args()


def enable_deterministic_cuda(seed: int | None = None) -> None:
    """Enable deterministic CUDA behavior for PyTorch."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if seed is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hidden_states(path: Path, num_samples: int) -> torch.Tensor:
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as f:
            hidden_states = f.get_tensor("hidden_states")
    else:
        hidden_states = torch.load(path)
    # hidden_states = hidden_states.float()
    if num_samples:
        hidden_states = hidden_states[:num_samples]
    return hidden_states


def load_alignment_matrix(
    data_dir: Path, embedding_model: str, method: str
) -> torch.Tensor | None:
    """Load the semantic alignment matrix if it exists."""
    align_path = data_dir / f"align_{method}_{embedding_model}.pt"
    if not align_path.exists():
        raise FileNotFoundError(
            f"Alignment matrix not found at {align_path}. "
            "Please run scripts/generate_alignment.py first."
        )
    print(f"Loading alignment matrix from {align_path}")
    W_align = torch.load(align_path)
    return W_align


def project_hidden_states_chunked(
    hidden_states: torch.Tensor,
    W_align: torch.Tensor,
    chunk_size: int = DEFAULT_PROJECTION_CHUNK_SIZE,
) -> torch.Tensor:
    """Project hidden states through the alignment matrix in chunks.

    Equivalent to ``F.linear(hidden_states, W_align.T)`` but with a progress bar
    and lower peak memory than materializing very large intermediate buffers.
    """
    n_samples = hidden_states.shape[0]
    weight = W_align.T
    out_dim = weight.shape[0]
    projected = torch.empty(
        n_samples,
        out_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for start in trange(0, n_samples, chunk_size, desc="Projecting hidden states"):
        end = min(start + chunk_size, n_samples)
        projected[start:end] = F.linear(hidden_states[start:end], weight)
    return projected


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
        max_no_improvement=25,
        batch_size=8192 * 8,
        max_iter=100,
        reassignment_ratio=0.001,
        verbose=1,
        device="cpu",
        dtype=hidden_states.dtype,
    )
    kmeans.fit(hidden_states)
    return kmeans.labels_, kmeans.cluster_centers_


def clean_clusters(
    all_labels: torch.Tensor,
    all_hidden_states: torch.Tensor,
    min_size: int,
    k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove tiny clusters and reindex labels to be contiguous.
    """
    device = all_labels.device
    label_counts = torch.bincount(all_labels, minlength=k)

    valid_mask = label_counts >= min_size
    valid_labels = torch.nonzero(valid_mask).squeeze(1)

    mapping = -torch.ones(k, dtype=torch.long, device=device)
    mapping[valid_labels] = torch.arange(valid_labels.size(0), device=device)

    total = all_labels.shape[0]
    selected_idx_list: list[torch.Tensor] = []
    remapped_list: list[torch.Tensor] = []

    for start in trange(0, total, CLEAN_CHUNK_SIZE):
        end = min(start + CLEAN_CHUNK_SIZE, total)
        chunk_labels = all_labels[start:end]
        remapped_chunk = mapping[chunk_labels]
        valid_mask = remapped_chunk != -1
        if valid_mask.any():
            local_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            selected_idx_list.append(local_idx + start)
            remapped_list.append(remapped_chunk[valid_mask])

    if not selected_idx_list:
        empty_labels = torch.empty((0,), dtype=torch.long, device=device)
        empty_states = all_hidden_states[:0]
        empty_idx = torch.empty((0,), dtype=torch.long, device=device)
        return empty_labels, empty_states, empty_idx

    selected_idx = torch.cat(selected_idx_list)
    remapped_labels = torch.cat(remapped_list)
    return remapped_labels, all_hidden_states[selected_idx], selected_idx


def balanced_downsample(
    hidden: torch.Tensor, labels: torch.Tensor, max_per_class: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def balanced_downsample_cpu(hidden: torch.Tensor,
                            labels: torch.Tensor,
                            max_per_class: int,
                            num_classes: int = 256):
    """
    Caps each class at max_per_class. CPU-optimized:
    - O(N) early-exit via bincount
    - O(N) downsample via shuffled scan (no argsort/unique)
    Returns: hidden_sub, labels_sub, selected_idx
    """
    assert labels.device.type == "cpu", "This version is CPU-focused."
    assert labels.dtype in (torch.int64, torch.int32), "labels must be integer type."

    N = labels.numel()

    # O(N) counts
    counts = torch.bincount(labels, minlength=num_classes)
    if int(counts.max()) <= max_per_class:
        idx = torch.arange(N, dtype=torch.long)
        return hidden, labels, idx

    # O(N) shuffle + take up to cap per class
    perm = torch.randperm(N)  # random order of examples
    taken = torch.zeros(num_classes, dtype=torch.int32)

    selected = torch.empty(N, dtype=torch.long)  # upper bound
    out_n = 0

    # Python loop over 1.5M is usually fine on CPU; it’s linear and avoids sorting.
    # If you want even faster, we can do a vectorized / chunked variant.
    for i in perm.tolist():
        c = int(labels[i])
        if taken[c] < max_per_class:
            selected[out_n] = i
            out_n += 1
            taken[c] += 1

    selected = selected[:out_n]
    return hidden[selected], labels[selected], selected



def ridge_regression(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> torch.Tensor:
    d_in = X.shape[1]
    I = torch.eye(d_in, device=X.device)
    return torch.linalg.solve(X.T @ X + alpha * I, X.T @ Y)

# cholesky-based ridge regression for better numerical stability


def ridge_regression_cholesky(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> torch.Tensor:
    d = X.shape[1]
    I = torch.eye(d, device=X.device, dtype=X.dtype)
    A = X.T @ X
    A = A + alpha * I
    b = X.T @ Y
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(b, L)


@torch.no_grad()
def ridge_regression_chunked(X: torch.Tensor, Y: torch.Tensor, alpha: float, chunk: int = 100_000) -> torch.Tensor:
    """
    X: [N, d_in]
    Y: [N, k]
    Returns W: [d_in, k]
    """
    N, d_in = X.shape
    k = Y.shape[1]

    XtX = torch.zeros((d_in, d_in), device=X.device, dtype=torch.float64)
    XtY = torch.zeros((d_in, k), device=X.device, dtype=torch.float64)

    for i in trange(0, N, chunk):
        Xb = X[i:i+chunk].to(torch.float64)
        Yb = Y[i:i+chunk].to(torch.float64)
        XtX.addmm_(Xb.T, Xb)
        XtY.addmm_(Xb.T, Yb)

    XtX.diagonal().add_(alpha)
    W = torch.linalg.solve(XtX, XtY)
    return W.to(X.dtype)


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
    plt.hist(values.cpu().float().numpy().flatten(), density=True,
             bins=bins, alpha=0.7, color="blue")
    plt.title("Projections Histogram")
    plt.xlabel("Projection Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _log_phase(phase: str, start: float) -> float:
    now = time.perf_counter()
    elapsed = now - start
    print(f"[timing] {phase}: {elapsed:.2f}s")
    return now


def train(train_hidden_states, train_hidden_states_orig, train_labels, k, W_align=None):
    # Downsample to cap per-class size for efficiency.
    print("Downsampling training data...")
    train_hidden_states_ds, train_labels_ds, selected_idx = balanced_downsample_cpu(
        train_hidden_states, train_labels, max_per_class=MAX_PER_CLASS,
        num_classes=k
    )

    print(
        f"Train size: {len(train_hidden_states)} ({len(train_hidden_states_ds)} after downsample), "
    )
    print("Training ridge regression selector...")
    # Train ridge regression selector.
    train_labels_one_hot = F.one_hot(train_labels_ds, num_classes=k).float()
    W_selector = ridge_regression_chunked(
        train_hidden_states_ds,
        train_labels_one_hot,
        alpha=RIDGE_ALPHA,
    )
    if torch.isnan(W_selector).any():
        raise ValueError("NaNs found in regression weights.")

    final_matrix = W_selector.T

    if W_align is not None:
        final_matrix = final_matrix @ W_align.T

    if RESCALE_MATRIX:
        print("rescaling selector matrix...")
        train_hidden_states_orig_ds = train_hidden_states_orig[selected_idx]
        final_matrix = rescale(
            final_matrix, k, train_hidden_states_orig_ds, train_labels_ds
        )

    if torch.isnan(final_matrix).any():
        raise ValueError("NaNs found after post processing.")
    return final_matrix


def main() -> None:
    args = parse_args()
    enable_deterministic_cuda(args.seed)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.perf_counter()

    dataset_suffix = args.dataset_name.split("/")[-1]
    model_suffix = args.model_name.split("/")[-1]
    data_dir = Path(args.root_dir) / "data" / \
        f"{dataset_suffix}_{model_suffix}"
    base_model_dir = (
        Path(args.selector_dir)
        if args.selector_dir
        else Path(args.root_dir) / "saved_models_new" / f"{dataset_suffix}_{model_suffix}"
    )
    if args.hidden_states_format == "safetensors":
        hidden_states_path = data_dir / HIDDEN_STATES_SAFETENSORS_FILENAME
    elif args.hidden_states_format == "pt":
        hidden_states_path = data_dir / HIDDEN_STATES_FILENAME
    else:
        safetensors_path = data_dir / HIDDEN_STATES_SAFETENSORS_FILENAME
        hidden_states_path = (
            safetensors_path if safetensors_path.exists() else data_dir / HIDDEN_STATES_FILENAME
        )
    plot_dir = Path(args.root_dir) / PLOT_DIR_NAME / \
        f"{dataset_suffix}_{model_suffix}" / f"k_{args.k}"

    hidden_states = load_hidden_states(hidden_states_path, args.num_samples)
    t0 = _log_phase("load_hidden_states", t0)
    print(f"Loaded hidden states: {hidden_states.shape}")
    # Shuffle with seed
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    perm = torch.randperm(hidden_states.size(0), generator=generator)
    hidden_states = hidden_states[perm]


    # Load alignment matrix if semantic alignment is enabled
    W_align = None
    hidden_states_for_clustering = hidden_states
    if args.sem_align:
        W_align = load_alignment_matrix(
            data_dir, args.embedding_model, args.align_method
        ).to(dtype=hidden_states.dtype)
        print(f"Alignment matrix shape: {W_align.shape}")
        print(f"Projecting in chunks of {args.projection_chunk_size:,} rows...")
        hidden_states_for_clustering = project_hidden_states_chunked(
            hidden_states,
            W_align,
            chunk_size=args.projection_chunk_size,
        )
        print(
            f"Projected hidden states shape: {hidden_states_for_clustering.shape}")
        t0 = _log_phase("semantic_alignment_projection", t0)

    l2_norm = hidden_states_for_clustering.norm(p=2, dim=1)
    norm_mean = l2_norm.mean().item()
    norm_std = l2_norm.std().item()
    print(f"L2 Norm Mean: {norm_mean:.4f}, L2 Norm Std: {norm_std:.4f}")
    t0 = _log_phase("l2_norm_stats", t0)

    prf_key = args.prf_key % 2**64
    if args.labeling == "kmeans":
        if args.sem_align:
            hidden_states_for_clustering = F.normalize(
                hidden_states_for_clustering, dim=1)

        all_labels, _ = generate_kmeans_labels(
            hidden_states_for_clustering, prf_key, args.k)
        t0 = _log_phase("kmeans_labeling", t0)
        print("Cleaning clusters...")
        all_labels_cleaned, hidden_states_for_clustering_cleaned, valid_idx = clean_clusters(
            all_labels,
            hidden_states_for_clustering,
            min_size=MIN_CLUSTER_SIZE,
            k=args.k
        )
        t0 = _log_phase("clean_clusters", t0)
        print("Syncing original hidden states with cleaned labels...")
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
        t0 = _log_phase("rand_proj_labeling", t0)

    unique, counts = torch.unique(all_labels, return_counts=True)
    counts_np = counts.cpu().numpy()
    semalign_prefix = (
        f"semalign_{args.align_method}_{args.embedding_model}_"
        if args.sem_align
        else ""
    )
    # save_pie_chart(counts_np, unique.cpu().numpy(),
    #                plot_dir / f"{semalign_prefix}cluster_distribution_{args.labeling}_k{args.k}.png")
    print(f"Cluster counts: {counts_np.tolist()}")

    train_idx, val_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=0.25,
        stratify=all_labels.cpu().numpy(),
        random_state=args.seed,
    )
    t0 = _log_phase("train_val_split", t0)
    train_hidden_states = hidden_states_for_clustering[train_idx]
    train_hidden_states_orig = hidden_states[train_idx]
    train_labels = all_labels[train_idx]
    val_hidden_states = hidden_states[val_idx]
    val_labels = all_labels[val_idx]

    final_matrix = train(
        train_hidden_states, train_hidden_states_orig, train_labels, k,
        W_align=W_align
    )
    t0 = _log_phase("train_selector_matrix", t0)

    train_acc = compute_accuracy(
        final_matrix, train_hidden_states_orig, train_labels)
    val_acc = compute_accuracy(final_matrix, val_hidden_states, val_labels)
    t0 = _log_phase("compute_accuracy", t0)

    logits_val = F.linear(val_hidden_states, final_matrix)
    one_hotness_metrics = measure_one_hotness(
        logits_val, labels=val_labels, p=1)
    t0 = _log_phase("one_hotness_metrics", t0)

    watermark_matrix = OpenStamp._make_watermarking_matrix(
        32000, 1, 0.25, seed=prf_key, n_clusters=k
    )
    watermark_matrix = watermark_matrix.to(final_matrix.device).to(final_matrix.dtype)
    delta_w = watermark_matrix @ final_matrix
    projections = F.linear(
        val_hidden_states[:PROJECTION_EVAL_SAMPLES], delta_w)
    save_histogram(projections, plot_dir /
                   f"{semalign_prefix}projections_hist.png")
    t0 = _log_phase("watermark_projection_hist", t0)

    extreme_threshold = 3.0
    extreme_counts = (projections.abs() > extreme_threshold).sum(dim=1)
    extreme_fraction = extreme_counts.float().mean().item() / projections.size(1)

    semalign_suffix = (
        f"_semalign_{args.align_method}_{args.embedding_model}"
        if args.sem_align
        else ""
    )
    model_dir = Path(f"{base_model_dir}_k{k}{semalign_suffix}")
    model_dir.mkdir(parents=True, exist_ok=True)
    selector_path = model_dir / "selector_matrix.pth"
    torch.save(final_matrix, selector_path)
    print(f"Saved selector matrix to {selector_path}")
    if args.sem_align:
        print(
            "Note: This matrix is composed (selector @ alignment) "
            "and maps directly from hidden states to labels."
        )
    t0 = _log_phase("save_selector_matrix", t0)

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
        "align_method": args.align_method if args.sem_align else None,
    }

    metrics_path = model_dir / "selector_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))
    _log_phase("save_metrics", t0)


if __name__ == "__main__":
    main()