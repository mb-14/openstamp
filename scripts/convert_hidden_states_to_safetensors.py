#!/usr/bin/env python
"""Convert hidden_states.pt files to safetensors format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


HIDDEN_STATES_FILENAME = "hidden_states.pt"
SAFETENSORS_FILENAME = "hidden_states.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert hidden_states.pt files to safetensors."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="Root directory to search for hidden_states.pt files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing .safetensors files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files to convert without writing output.",
    )
    return parser.parse_args()


def _load_hidden_states(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict) and "hidden_states" in obj:
        tensor = obj["hidden_states"]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("hidden_states in dict is not a torch.Tensor")
        return tensor
    raise TypeError(f"Unsupported format in {path}")


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)

    candidates = sorted(root_dir.rglob(HIDDEN_STATES_FILENAME))
    if not candidates:
        print("No hidden_states.pt files found.")
        return

    for path in candidates:
        out_path = path.with_name(SAFETENSORS_FILENAME)
        if out_path.exists() and not args.overwrite:
            print(f"Skipping (exists): {out_path}")
            continue

        print(f"Converting: {path} -> {out_path}")
        if args.dry_run:
            continue

        hidden_states = _load_hidden_states(path)
        save_file({"hidden_states": hidden_states}, out_path)


if __name__ == "__main__":
    main()
