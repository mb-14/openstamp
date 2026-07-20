#!/usr/bin/env python3
"""Plot finetuning resistance curves (TPR@1%FPR vs step) for COLM figures.

Includes OpenStamp, GaussMark, KGW Distilled, and Unremovable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerLine2D

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from _plot_style import METHOD_COLORS, add_tex_flag, apply_paper_style, save_figure

METHODS = ("openstamp", "gaussmark", "distilled", "unremovable")

# Legend order: open-weight methods first, then decoding-based OpenStamp.
PLOT_ORDER = ("GaussMark", "Unremovable", "KGW Distilled", "OpenStamp")

MODELS = {
    "llama": {
        "plot_name": "finetuning_llama",
        "csv_suffix": "llama",
    },
    "mistral": {
        "plot_name": "finetuning_mistral",
        "csv_suffix": "mistral",
    },
}

METHOD_TO_LABEL = {
    "openstamp": "OpenStamp",
    "gaussmark": "GaussMark",
    "distilled": "KGW Distilled",
    "unremovable": "Unremovable",
}


def load_aggregated(data_dir: Path, model_suffix: str) -> pd.DataFrame:
    frames = []
    for method in METHODS:
        path = data_dir / f"{method}_{model_suffix}_ft.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing finetuning CSV: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def plot_finetuning(
    df: pd.DataFrame,
    *,
    plot_name: str,
    out_dir: Path,
    use_tex: bool,
) -> list[Path]:
    use_tex = apply_paper_style(use_tex=use_tex)
    df_plot = df.copy()
    df_plot["label"] = df_plot["method"].map(METHOD_TO_LABEL)

    fig, ax = plt.subplots()
    lines, labels = [], []
    for label in PLOT_ORDER:
        sub = df_plot[df_plot["label"] == label].sort_values("step")
        if sub.empty:
            continue
        legend_label = r"\textbf{OpenStamp}" if (use_tex and label == "OpenStamp") else label
        line = ax.errorbar(
            sub["step"],
            sub["tpr_1_fpr_mean"],
            yerr=sub["tpr_1_fpr_std"],
            label=legend_label,
            color=METHOD_COLORS[label],
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            linestyle="-",
            capsize=2,
            elinewidth=1.0,
        )
        lines.append(line[0])
        labels.append(legend_label)

    ylabel = r"TPR@1\%FPR" if use_tex else "TPR@1%FPR"
    ax.set_xlabel("Finetuning Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    legend = ax.legend(
        lines,
        labels,
        loc="upper right",
        frameon=True,
        handler_map={plt.Line2D: HandlerLine2D(numpoints=0)},
    )
    if not use_tex:
        for text in legend.get_texts():
            if "OpenStamp" in text.get_text():
                text.set_fontweight("semibold")

    fig.tight_layout()
    paths = save_figure(fig, out_dir, plot_name)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DIR / "finetuning",
        help="Directory with {method}_{llama|mistral}_ft.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DIR / "plots",
        help="Directory for PDF/PNG outputs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
        help="Which model families to plot",
    )
    add_tex_flag(parser)
    args = parser.parse_args()

    for name in args.models:
        cfg = MODELS[name]
        df = load_aggregated(args.data_dir, cfg["csv_suffix"])
        print(f"{name}: {df.groupby('method').size().to_dict()}")
        paths = plot_finetuning(
            df,
            plot_name=cfg["plot_name"],
            out_dir=args.out_dir,
            use_tex=args.tex,
        )
        for p in paths:
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
