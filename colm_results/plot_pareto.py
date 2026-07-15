#!/usr/bin/env python3
"""Plot PPL vs TPR@1%FPR Pareto tradeoff (COLM ppl_tpr_plot)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from _plot_style import (
    DECODING_COLOR,
    OPEN_WEIGHT_COLOR,
    add_tex_flag,
    apply_paper_style,
    save_figure,
)

# Paper legend / draw order.
LABEL_ORDER = ["OpenStamp", "KGW Distilled", "Gaussmark", "Unremovable", "KGW", "KGW + LLR"]
MARKER_MAP = {
    "OpenStamp": "*",
    "KGW Distilled": "v",
    "Gaussmark": "D",
    "Unremovable": "X",
    "KGW": "s",
    "KGW + LLR": "o",
}
OPEN_WEIGHT = {"OpenStamp", "KGW Distilled", "Gaussmark", "Unremovable"}

METHOD_DISPLAY = {
    "OpenStamp": "OpenStamp",
    "KGW Distilled": "KGW Distilled",
    "GaussMark": "Gaussmark",
    "Unremovable": "Unremovable",
    "KGW": "KGW",
    "KGW + LLR": "KGW + LLR",
}


def pareto_frontier(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    """Lower PPL is better, higher TPR is better → non-dominated frontier."""
    data = sorted(zip(x.tolist(), y.tolist()), key=lambda pair: (pair[0], -pair[1]))
    frontier: list[tuple[float, float]] = []
    max_y = -np.inf
    for xi, yi in data:
        if yi >= max_y:
            frontier.append((xi, yi))
            max_y = yi
    if not frontier:
        return [], []
    px, py = zip(*frontier)
    return list(px), list(py)


def load_points(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Method", "PPL", "TPR@1%FPR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["label"] = df["Method"].map(METHOD_DISPLAY)
    unknown = df[df["label"].isna()]["Method"].unique().tolist()
    if unknown:
        raise ValueError(f"Unknown Method values in CSV: {unknown}")
    df["ppl"] = df["PPL"].astype(float)
    df["tpr"] = df["TPR@1%FPR"].astype(float)
    return df


def plot_pareto(df: pd.DataFrame, *, out_dir: Path, use_tex: bool, stem: str) -> list[Path]:
    use_tex = apply_paper_style(use_tex=use_tex)

    color_map = {
        label: OPEN_WEIGHT_COLOR if label in OPEN_WEIGHT else DECODING_COLOR
        for label in LABEL_ORDER
    }

    px, py = pareto_frontier(df["ppl"].to_numpy(), df["tpr"].to_numpy())

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    for label in LABEL_ORDER:
        group = df[df["label"] == label]
        for _, row in group.iterrows():
            size = 120 if label == "OpenStamp" else 60
            ax.scatter(
                row["ppl"],
                row["tpr"],
                s=size,
                color=color_map[label],
                marker=MARKER_MAP[label],
                linewidth=0.6,
                alpha=0.65,
                edgecolor="black",
                zorder=2,
            )

    if px:
        ax.plot(px, py, color="dimgray", linestyle="--", linewidth=1.3, zorder=0)

    # Match paper notebook labels (arrows kept as unicode even under TeX).
    ax.set_xlabel("PPL (↓)", fontsize=10)
    ax.set_ylabel(r"TPR@1\%FPR (↑)" if use_tex else "TPR@1%FPR (↑)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    custom_lines = []
    for label in LABEL_ORDER:
        legend_label = (
            r"\textbf{OpenStamp}" if (use_tex and label == "OpenStamp") else label
        )
        custom_lines.append(
            Line2D(
                [0],
                [0],
                marker=MARKER_MAP[label],
                label=legend_label,
                markerfacecolor=(*color_map[label][:3], 0.65),
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=14 if label == "OpenStamp" else 8.5,
                linestyle="None",
            )
        )
    legend = ax.legend(
        handles=custom_lines,
        fontsize=8.5,
        loc="lower right",
        frameon=False,
        labelspacing=0.8,
    )
    if not use_tex:
        for text in legend.get_texts():
            if "OpenStamp" in text.get_text():
                text.set_fontweight("semibold")

    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DIR / "pareto" / "pareto_ppl_tpr_points.csv",
        help="Pareto sweep CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DIR / "plots",
        help="Directory for PDF/PNG outputs",
    )
    parser.add_argument(
        "--stem",
        default="ppl_tpr_plot_llama",
        help="Output filename stem (default: ppl_tpr_plot_llama)",
    )
    add_tex_flag(parser)
    args = parser.parse_args()

    df = load_points(args.csv)
    print(f"loaded {len(df)} points from {args.csv}")
    print(df.groupby("label").size().to_dict())
    paths = plot_pareto(df, out_dir=args.out_dir, use_tex=args.tex, stem=args.stem)
    for p in paths:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
