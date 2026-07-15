#!/usr/bin/env python3
"""Plot relative downstream accuracy vs unwatermarked baseline (COLM figures)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from _plot_style import add_tex_flag, apply_paper_style, save_figure

BENCHMARKS = ("arc_challenge", "boolq", "hellaswag")
BENCHMARK_LABELS = {
    "arc_challenge": "ARC-C",
    "boolq": "BoolQ",
    "hellaswag": "HellaSwag",
}

# CSV model suffix -> figure label.
METHOD_LABEL = {
    "OpenStamp": "OpenStamp",
    "GaussMark": "Gaussmark",
    "Distilled": "KGW Distilled",
}

# Paper x-axis order (left → right).
METHOD_ORDER = ("OpenStamp", "Gaussmark", "KGW Distilled")

FAMILIES = (
    ("Llama-2-7B", "relative_ds_accuracy_llama"),
    ("Mistral-7B", "relative_ds_accuracy_mistral"),
)


def parse_model_cell(cell: str) -> tuple[str, str] | None:
    """Split 'Llama-2-7B OpenStamp' into (family, method_key)."""
    m = re.match(r"^(Llama-2-7B|Mistral-7B)\s+(.+)$", cell.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


def load_relative_accuracies(csv_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """
    Returns nested dict: family -> method_label -> benchmark -> relative %.

    Relative accuracy is capped at 100%, matching the paper notebook.
    """
    df = pd.read_csv(csv_path)
    scores: dict[tuple[str, str, str], float] = {}
    for _, row in df.iterrows():
        parsed = parse_model_cell(str(row["model"]))
        if parsed is None:
            continue
        family, method_key = parsed
        bench = str(row["benchmark"])
        if bench not in BENCHMARKS:
            continue
        scores[(family, method_key, bench)] = float(row["score"])

    relative: dict[str, dict[str, dict[str, float]]] = {}
    for family, _ in FAMILIES:
        relative[family] = {}
        for method_key, label in METHOD_LABEL.items():
            method_scores: dict[str, float] = {}
            for bench in BENCHMARKS:
                base = scores.get((family, "Baseline", bench))
                wm = scores.get((family, method_key, bench))
                if base is None or wm is None or base == 0:
                    continue
                method_scores[bench] = min(100.0 * wm / base, 100.0)
            if method_scores:
                relative[family][label] = method_scores
    return relative


def plot_family(
    method_scores: dict[str, dict[str, float]],
    *,
    stem: str,
    out_dir: Path,
    use_tex: bool,
) -> list[Path]:
    use_tex = apply_paper_style(use_tex=use_tex)

    method_labels = [m for m in METHOD_ORDER if m in method_scores]
    x = np.arange(len(method_labels))
    width = 0.25

    cmap = plt.colormaps["tab10"].resampled(len(BENCHMARKS))
    colors = [mcolors.to_rgba(cmap(i), alpha=0.7) for i in range(len(BENCHMARKS))]

    fig, ax = plt.subplots()
    for i, benchmark in enumerate(BENCHMARKS):
        for j, label in enumerate(method_labels):
            value = method_scores[label].get(benchmark, 0.0)
            bar_label = BENCHMARK_LABELS[benchmark] if j == 0 else None
            ax.bar(
                x[j] + i * width,
                value,
                width,
                color=colors[i],
                edgecolor="black",
                linewidth=0.4,
                label=bar_label,
            )

    ylabel = r"Relative Accuracy (\%)" if use_tex else "Relative Accuracy (%)"
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(80, 105)
    ax.set_yticks([80, 85, 90, 95, 100])
    ax.tick_params(axis="y", labelsize=8)
    ax.axhline(100, color="black", linestyle="--", linewidth=0.7)

    ax.set_xticks(x + width * (len(BENCHMARKS) - 1) / 2)
    tick_labels = [
        r"\textbf{OpenStamp}" if (use_tex and label == "OpenStamp") else label
        for label in method_labels
    ]
    ax.set_xticklabels(tick_labels, fontsize=9)
    if not use_tex:
        for tick, label in zip(ax.get_xticklabels(), method_labels):
            if label == "OpenStamp":
                tick.set_fontweight(600)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(BENCHMARKS),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DIR / "downstream" / "downstream_evals.csv",
        help="Aggregated downstream eval CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DIR / "plots",
        help="Directory for PDF/PNG outputs",
    )
    add_tex_flag(parser)
    args = parser.parse_args()

    relative = load_relative_accuracies(args.csv)
    for family, stem in FAMILIES:
        method_scores = relative.get(family, {})
        if not method_scores:
            print(f"skip {family}: no scores")
            continue
        print(f"{family}:")
        for label in METHOD_ORDER:
            if label not in method_scores:
                continue
            benches = method_scores[label]
            parts = ", ".join(f"{BENCHMARK_LABELS[b]}={v:.1f}%" for b, v in benches.items())
            print(f"  {label}: {parts}")
        paths = plot_family(
            method_scores,
            stem=stem,
            out_dir=args.out_dir,
            use_tex=args.tex,
        )
        for p in paths:
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
