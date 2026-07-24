#!/usr/bin/env python3
"""Evaluate watermark detection metrics on generated samples."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer

from src.christmark import ChristMark
from src.gaussmark import GaussMark
from src.kgw_distilled import KGWDistilled
from src.kgwmark import KGWMark
from src.openstamp import OpenStamp
from src.rlmark import RLMark
from src.unigramwm import Unigram
from src.adaptive_watermark import AdaptiveMark
from src.utils import load_model

torch.manual_seed(42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score generated samples and write detection metrics."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.getenv("OUTPUT_FILE"),
        help="Path to the sample JSON (also reads OUTPUT_FILE env var).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for saved plots (default: <output_file parent>/logs).",
    )
    return parser.parse_args()


def compute_metrics(watermark_scores: np.ndarray, null_scores: np.ndarray) -> dict:
    watermark_scores = np.asarray(watermark_scores, dtype=np.float64)
    null_scores = np.asarray(null_scores, dtype=np.float64)

    min_sweep = min(watermark_scores.min(), null_scores.min()) - 1
    max_sweep = max(watermark_scores.max(), null_scores.max()) + 1

    y_true = np.concatenate([
        np.ones_like(watermark_scores),
        np.zeros_like(null_scores),
    ])
    y_score = np.concatenate([watermark_scores, null_scores])
    auroc = roc_auc_score(y_true, y_score)

    f1_scores = []
    thresholds = np.linspace(min_sweep, max_sweep, 1000)
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores.append(f1_score)

    f1_scores = np.array(f1_scores)
    thresholds = np.array(thresholds)
    best_f1_score = f1_scores.max()
    best_indices = np.where(f1_scores == best_f1_score)[0]

    best_precisions = []
    for idx in best_indices:
        threshold = thresholds[idx]
        y_pred = (y_score >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        best_precisions.append(precision)

    best_precisions = np.array(best_precisions)
    max_precision = best_precisions.max()
    precision_indices = best_indices[np.where(best_precisions == max_precision)[0]]
    best_threshold = thresholds[precision_indices].max()

    fpr_array, tpr_array, roc_thresholds = roc_curve(y_true, y_score)
    tpr_interp = interp1d(
        fpr_array,
        tpr_array,
        kind="linear",
        bounds_error=False,
        fill_value=(tpr_array[0], tpr_array[-1]),
    )
    tpr_at_1_fpr = float(tpr_interp(0.01))
    tpr_at_01_fpr = float(tpr_interp(0.001))
    tpr_at_001_fpr = float(tpr_interp(0.0001))
    tpr_at_005_fpr = float(tpr_interp(0.0005))
    tpr_at_5_fpr = float(tpr_interp(0.05))

    fpr_diff = np.abs(fpr_array - 0.01)
    threshold_at_1_fpr = float(roc_thresholds[fpr_diff.argmin()])

    return {
        "auroc": float(auroc),
        "best_f1_score": float(best_f1_score),
        "tpr_1_fpr": tpr_at_1_fpr,
        "tpr_0.1_fpr": tpr_at_01_fpr,
        "tpr_0.05_fpr": tpr_at_005_fpr,
        "tpr_0.01_fpr": tpr_at_001_fpr,
        "tpr_5_fpr": tpr_at_5_fpr,
        "best_threshold": float(best_threshold),
        "threshold_at_1_fpr": threshold_at_1_fpr,
    }


def build_watermark(output_data: dict, model, tokenizer):
    watermark_type = output_data["watermark"]
    config = output_data["config"]
    batch_size = 64

    if watermark_type == "gaussmark":
        watermark = GaussMark(
            sigma=config["sigma"],
            tokenizer=tokenizer,
            model=model,
            seed=config["watermark_seed"],
            target_param_name=config["target_param_name"],
        )
        batch_size = 8
    elif watermark_type in {"unremovable", "christ"}:
        watermark = ChristMark(
            epsilon=config["epsilon"],
            seed=config["watermark_seed"],
            tokenizer=tokenizer,
            model=None,
            vocab_size=config.get("vocab_size"),
        )
    elif watermark_type in ["openstamp", "openstamp_binom", "openstamp_discrete"]:
        selector_matrix_dir = config.get("selector_matrix_dir")
        if not selector_matrix_dir:
            raise ValueError(
                "Selector matrix directory not provided in config for mb watermark."
            )
        final_matrix_path = os.path.join(selector_matrix_dir, "selector_matrix.pth")
        final_weight = torch.load(final_matrix_path)
        if watermark_type == "openstamp":
            detection_type = "llr"
        elif watermark_type == "openstamp_binom":
            detection_type = "binom"
        else:
            detection_type = "llr_discrete"
        watermark = OpenStamp.from_config(
            delta=config["delta"],
            gamma=config["gamma"],
            seed=config["watermark_seed"],
            final_weight=final_weight,
            model=model,
            tokenizer=tokenizer,
            unembedding_param_name=config["unembedding_param_name"],
            detection_type=detection_type,
        )
    elif watermark_type == "noise":
        watermark = OpenStamp.noise_injection(
            delta=config["delta"],
            seed=config["watermark_seed"],
            model=model,
            tokenizer=tokenizer,
            unembedding_param_name=config["unembedding_param_name"],
            distribution=config["distribution"],
        )
    elif watermark_type == "distilled":
        watermark = KGWDistilled(
            gamma=config["gamma"],
            delta=config["delta"],
            seeding_scheme=config["seeding_scheme"],
            hash_key=config["watermark_seed"],
            kgw_device=config["kgw_device"],
            tokenizer=tokenizer,
        )
    elif watermark_type in ["kgw", "kgw_llr"]:
        watermark = KGWMark(
            gamma=config["gamma"],
            delta=config["delta"],
            hash_key=config["watermark_seed"],
            kgw_device=config["kgw_device"],
            model=model,
            tokenizer=tokenizer,
            llr_detection=watermark_type == "kgw_llr",
        )
    elif watermark_type == "unigram":
        watermark = Unigram(
            gamma=config["gamma"],
            delta=config["delta"],
            hash_key=config["watermark_seed"],
            tokenizer=tokenizer,
        )
    elif watermark_type == "adaptive":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        watermark = AdaptiveMark(
            tokenizer=tokenizer,
            model=None,
            device=device,
            prompt_length=config.get("prompt_length", 50),
            delta=config["delta"],
            mapping_seed=config.get("watermark_seed", 66),
        )
        batch_size = 16
    elif watermark_type == "rl":
        watermark = RLMark(
            rl_model_path=config["rl_model_path"],
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unsupported watermark type: {watermark_type}")

    return watermark, batch_size


def get_scores(watermark, samples: dict, column: str, batch_size: int) -> torch.Tensor:
    all_scores = []
    data = samples[column]
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        with torch.no_grad():
            scores = watermark.score_text_batch(batch)
            all_scores.append(scores)
    return torch.cat(all_scores)


def save_score_histogram(
    negative_z: torch.Tensor,
    positive_z: torch.Tensor,
    column: str,
    log_dir: str,
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    plot_path = os.path.join(log_dir, f"hist_{column}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        negative_z.cpu().numpy(),
        bins=50,
        alpha=0.5,
        label="Human Text",
        color="blue",
    )
    ax.hist(
        positive_z.cpu().numpy(),
        bins=50,
        alpha=0.5,
        label="Watermarked Text",
        color="orange",
    )
    ax.set_title("Avg LLR Scores")
    ax.set_xlabel("Avg LLR Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def compute_scores(
    watermark,
    samples: dict,
    column: str,
    negative_z: torch.Tensor,
    batch_size: int,
    log_dir: str,
) -> dict:
    positive_z = get_scores(watermark, samples, column, batch_size)
    mean_positive_z = positive_z.mean().item()
    std_positive_z = positive_z.std().item()

    save_score_histogram(negative_z, positive_z, column, log_dir)

    print(f"Mean positive z value: {mean_positive_z}")
    print(f"Std positive z value: {std_positive_z}")

    watermark_scores = positive_z.cpu().numpy()
    null_scores = negative_z.cpu().numpy()
    return compute_metrics(watermark_scores, null_scores)


def main() -> None:
    args = parse_args()
    if not args.output_file:
        raise SystemExit(
            "Please set --output_file or the OUTPUT_FILE environment variable."
        )

    log_dir = args.log_dir or os.path.join(os.path.dirname(args.output_file), "logs")
    os.makedirs(log_dir, exist_ok=True)

    batch_size = 64
    print(f"Batch size: {batch_size}")

    with open(args.output_file, "r", encoding="utf-8") as handle:
        output_data = json.load(handle)

    samples = output_data["samples"]
    model_name = output_data["model_name"]
    print(f"Model name: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    watermark_type = output_data["watermark"]
    should_load_model = watermark_type not in [
        "distilled",
        "kgw",
        "unigram",
        "christ",
        "unremovable",
        "adaptive",
    ]
    if should_load_model:
        model, tokenizer = load_model(model_name)
    else:
        model = None

    watermark, batch_size = build_watermark(output_data, model, tokenizer)
    if watermark_type == "gaussmark":
        print(f"Batch size: {batch_size}")

    negative_z = get_scores(watermark, samples, "human_text", batch_size)
    mean_negative_z = negative_z.mean().item()
    std_negative_z = negative_z.std().item()
    print(f"Mean negative z value: {mean_negative_z}")
    print(f"Std negative z value: {std_negative_z}")

    metrics = compute_scores(
        watermark,
        samples,
        "model_text",
        negative_z,
        batch_size,
        log_dir,
    )
    output_data["metrics"] = metrics
    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    with torch.no_grad():
        torch.cuda.empty_cache()

    batch_size = max(1, batch_size // 2)

    optional_columns = [
        ("dipper_text_lex60_order0", "metrics_dipper_text_lex60_order0", "Dipper 60 Metrics"),
        ("dipper_text_lex20_order0", "metrics_dipper_text_lex20_order0", "Dipper 20 Metrics"),
        ("llm_paraphrase", "metrics_llm_paraphrase", "LLM Paraphrase Metrics"),
    ]
    for column, metrics_key, label in optional_columns:
        if column not in samples:
            continue
        metrics = compute_scores(
            watermark,
            samples,
            column,
            negative_z,
            batch_size,
            log_dir,
        )
        output_data[metrics_key] = metrics
        print(f"{label}:")
        for key, value in metrics.items():
            print(f"    {key}: {value}")

    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=4)


if __name__ == "__main__":
    main()
