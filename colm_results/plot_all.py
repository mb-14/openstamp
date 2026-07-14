#!/usr/bin/env python3
"""Generate all COLM paper plots (finetuning, Pareto, downstream)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent

SCRIPTS = (
    "plot_finetuning.py",
    "plot_pareto.py",
    "plot_downstream.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DIR / "plots",
        help="Shared output directory for all plots",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tex",
        dest="tex",
        action="store_true",
        default=True,
        help="Render labels with LaTeX (default)",
    )
    group.add_argument(
        "--no-tex",
        dest="tex",
        action="store_false",
        help="Disable LaTeX text rendering",
    )
    args = parser.parse_args()

    for name in SCRIPTS:
        cmd = [sys.executable, str(_DIR / name), "--out-dir", str(args.out_dir)]
        cmd.append("--tex" if args.tex else "--no-tex")
        print(f"\n=== {name} ===", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
