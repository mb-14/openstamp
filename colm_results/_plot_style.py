"""Shared matplotlib styling for COLM paper figures."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

COLM_RESULTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = COLM_RESULTS_DIR.parent
AAAI_STYLE = REPO_ROOT / "notebooks" / "aaai.mplstyle"
DEFAULT_OUT_DIR = COLM_RESULTS_DIR / "plots"

# Open-weight methods (green) vs decoding-based (orange), matching paper figures.
OPEN_WEIGHT_COLOR = (0 / 255, 170 / 255, 110 / 255)
DECODING_COLOR = (1.0, 0.5, 0.0)

METHOD_COLORS = {
    "OpenStamp": "#2ca02c",
    "GaussMark": "#1f77b4",
    "Gaussmark": "#1f77b4",
    "Unremovable": "#9467bd",
    "KGW Distilled": "#ff7f0e",
}


def _tex_available() -> bool:
    if shutil.which("latex") is None:
        return False
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Palatino"],
            }
        )
        fig = plt.figure()
        fig.text(0.5, 0.5, r"probe")
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            fig.savefig(tmp.name)
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False


def apply_paper_style(*, use_tex: bool = True) -> bool:
    """Apply AAAI + Palatino style. Returns whether TeX rendering is active."""
    if AAAI_STYLE.is_file():
        plt.style.use(str(AAAI_STYLE))

    if use_tex and _tex_available():
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Palatino"],
            }
        )
        return True

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Palatino", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "dejavuserif",
        }
    )
    return False


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Save a PDF figure for the paper."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    return [pdf_path]


def add_tex_flag(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tex",
        dest="tex",
        action="store_true",
        default=True,
        help="Render labels with LaTeX (default; matches paper)",
    )
    group.add_argument(
        "--no-tex",
        dest="tex",
        action="store_false",
        help="Disable LaTeX text rendering",
    )
