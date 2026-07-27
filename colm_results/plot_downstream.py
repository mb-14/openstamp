#!/usr/bin/env python3
"""Plot relative downstream accuracy vs unwatermarked baseline (COLM figures)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from _plot_style import METHOD_COLORS, add_tex_flag, apply_paper_style, save_figure

BENCHMARKS = ("arc_challenge", "boolq", "hellaswag")
BENCHMARK_LABELS = {
    "arc_challenge": "ARC-C",
    "boolq": "BoolQ",
    "hellaswag": "HellaSwag",
}
# Same hues as the finetuning curves (GaussMark / KGW Distilled / OpenStamp).
BENCHMARK_STYLE = {
    "arc_challenge": {"color": METHOD_COLORS["GaussMark"]},
    "boolq": {"color": METHOD_COLORS["KGW Distilled"]},
    "hellaswag": {"color": METHOD_COLORS["OpenStamp"]},
}

# CSV model suffix -> figure label.
METHOD_LABEL = {
    "OpenStamp": "OpenStamp",
    "GaussMark": "Gaussmark",
    "Unremovable": "Unremovable",
    "Distilled": "KGW Distilled",
}

# Paper y-axis order (top → bottom).
METHOD_ORDER = ("Gaussmark", "Unremovable", "KGW Distilled", "OpenStamp")

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
    """Horizontal grouped bars: methods on y, relative % on x."""
    use_tex = apply_paper_style(use_tex=use_tex)

    method_labels = [m for m in METHOD_ORDER if m in method_scores]
    if method_labels:
        benches: list[str] = []
        for b in BENCHMARKS:
            if any(b in method_scores[m] for m in method_labels):
                benches.append(b)
    else:
        benches = []
    if not method_labels or not benches:
        raise ValueError(f"No scores to plot for {stem}")

    # Top → bottom matching METHOD_ORDER.
    y = np.arange(len(method_labels))[::-1]

    all_values = [
        method_scores[m][b]
        for m in method_labels
        for b in benches
        if b in method_scores[m]
    ]
    x_min = min(80.0, max(0.0, float(np.floor(min(all_values) / 5.0) * 5.0 - 5.0)))

    n_b = len(benches)
    # Bars flush within each method group (no gap between benchmarks).
    bar_h = min(0.78 / n_b, 0.30)
    offsets = (np.arange(n_b) - (n_b - 1) / 2.0) * bar_h

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Subtle alternating row bands to group methods.
    for i, yi in enumerate(y):
        if i % 2 == 0:
            ax.axhspan(yi - 0.42, yi + 0.42, color="#f5f5f5", zorder=0, lw=0)

    ax.axvline(100, color="#555555", linestyle="--", linewidth=0.9, zorder=1, alpha=0.85)

    for offset, benchmark in zip(offsets, benches):
        style = BENCHMARK_STYLE[benchmark]
        fill = style["color"]
        widths = []
        ys = []
        for i, label in enumerate(method_labels):
            if benchmark not in method_scores[label]:
                continue
            widths.append(method_scores[label][benchmark])
            ys.append(y[i] + offset)
        ax.barh(
            ys,
            widths,
            height=bar_h * 0.92,  # small gap between bars in a group
            color=fill,
            edgecolor=fill,
            linewidth=0.0,
            zorder=3,
            label=BENCHMARK_LABELS[benchmark],
        )

    ax.set_yticks(y)
    tick_labels = []
    for label in method_labels:
        if label == "OpenStamp":
            tick_labels.append(r"\textbf{OpenStamp}" if use_tex else "OpenStamp")
        else:
            tick_labels.append(label)
    ax.set_yticklabels(tick_labels, fontsize=8)
    if not use_tex:
        for tick, label in zip(ax.get_yticklabels(), method_labels):
            if label == "OpenStamp":
                tick.set_fontweight(600)

    ax.set_xlim(x_min, 101.5)
    ax.set_ylim(min(y) - 0.48, max(y) + 0.48)
    # Ticks up to 100 only; headroom past the baseline is visual only.
    xticks = [t for t in ax.get_xticks() if x_min <= t <= 100]
    if 100 not in xticks:
        xticks.append(100.0)
    ax.set_xticks(sorted(xticks))
    xlabel = r"Relative Accuracy (\%)" if use_tex else "Relative Accuracy (%)"
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(benches),
        frameon=False,
        fontsize=8,
        handletextpad=0.35,
        columnspacing=1.15,
        handlelength=1.15,
        borderaxespad=0.0,
    )

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
            parts = ", ".join(
                f"{BENCHMARK_LABELS[b]}={v:.1f}%" for b, v in benches.items()
            )
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
