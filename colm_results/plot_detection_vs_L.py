#!/usr/bin/env python3
"""Plot detection strength versus L for the COLM paper figure."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from _plot_style import METHOD_COLORS, add_tex_flag, apply_paper_style, save_figure

OPENSTAMP_LABEL = "OpenStamp"
REFERENCE_ORDER = ("GaussMark", "KGW Distilled", "Unremovable")
METHOD_DISPLAY = {
    "OpenStamp": "OpenStamp",
    "GaussMark": "Gaussmark",
    "KGW Distilled": "KGW Distilled",
    "Unremovable": "Unremovable",
}


def parse_mean(value: object) -> float:
    """Parse either a float or a 'mean+/-std' style CSV cell."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError("Cannot parse an empty numeric cell")
    match = re.match(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text)
    if match is None:
        raise ValueError(f"Cannot parse numeric mean from {value!r}")
    return float(match.group(1))


def load_detection_vs_l(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    required = {"Method", "L", "PPL_at_TPR>=0.90"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["method_label"] = df["Method"].map(METHOD_DISPLAY)
    unknown = df[df["method_label"].isna()]["Method"].unique().tolist()
    if unknown:
        raise ValueError(f"Unknown Method values in CSV: {unknown}")
    df["ppl_at_target"] = df["PPL_at_TPR>=0.90"].map(parse_mean)

    openstamp = df[df["Method"] == OPENSTAMP_LABEL].copy()
    if openstamp.empty:
        raise ValueError(f"{csv_path} has no {OPENSTAMP_LABEL} rows")
    openstamp["L"] = openstamp["L"].astype(int)
    openstamp = openstamp[openstamp["L"] < 300].sort_values("L")
    if openstamp.empty:
        raise ValueError(f"{csv_path} has no {OPENSTAMP_LABEL} rows with L < 300")

    references = df[df["Method"].isin(REFERENCE_ORDER)].copy()
    present = set(references["Method"])
    missing_refs = [method for method in REFERENCE_ORDER if method not in present]
    if missing_refs:
        raise ValueError(f"{csv_path} missing reference rows: {missing_refs}")
    references["_order"] = references["Method"].map(
        {method: idx for idx, method in enumerate(REFERENCE_ORDER)}
    )
    references = references.sort_values("_order")

    return openstamp, references


def plot_detection_vs_l(
    openstamp: pd.DataFrame,
    references: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str,
    use_tex: bool,
    x_max: float | None,
) -> list[Path]:
    use_tex = apply_paper_style(use_tex=use_tex)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(
        openstamp["L"],
        openstamp["ppl_at_target"],
        marker="o",
        markersize=4,
        linewidth=1.2,
        linestyle="-",
        color=METHOD_COLORS["OpenStamp"],
        label=r"\textbf{OpenStamp}" if use_tex else "OpenStamp",
        zorder=3,
    )

    for _, row in references.iterrows():
        method = row["Method"]
        ax.axhline(
            row["ppl_at_target"],
            color=METHOD_COLORS[method],
            linestyle="--",
            linewidth=1.0,
            label=row["method_label"],
            zorder=2,
        )

    ax.set_xlabel("L (number of lists)")
    # Match paper notebook / Pareto figure: lower PPL is better.
    ax.set_ylabel(
        r"PPL @ TPR $\geq$ 0.90 ($\downarrow$)" if use_tex else "PPL @ TPR ≥ 0.90 (↓)"
    )
    ax.set_xlim(left=0, right=x_max)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=8)

    if not use_tex:
        for text in ax.get_legend().get_texts():
            if text.get_text() == "OpenStamp":
                text.set_fontweight(600)

    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DIR / "detection_vs_L" / "detection_vs_L.csv",
        help="Detection-vs-L summary CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DIR / "plots",
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--stem",
        default="detection_vs_L",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional x-axis maximum. Defaults to matplotlib autoscaling.",
    )
    add_tex_flag(parser)
    args = parser.parse_args()

    openstamp, references = load_detection_vs_l(args.csv)
    print(f"loaded {len(openstamp)} OpenStamp points from {args.csv}")
    for _, row in references.iterrows():
        print(f"  {row['method_label']}: PPL={row['ppl_at_target']:.2f}")

    paths = plot_detection_vs_l(
        openstamp,
        references,
        out_dir=args.out_dir,
        stem=args.stem,
        use_tex=args.tex,
        x_max=args.x_max,
    )
    for path in paths:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
