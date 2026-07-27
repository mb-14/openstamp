from __future__ import annotations

from typing import Any

from datasets import load_dataset

from src.alpaca_split import load_alpaca_eval_dataset


def load_registry_dataset(spec: dict[str, Any], **kwargs: Any):
    """Load a dataset entry from the registry."""
    loader = spec.get("loader")
    if loader == "alpaca_eval":
        return load_alpaca_eval_dataset(
            repo_id=spec["path"],
            data_file=spec.get("data_file", "alpaca_eval.json"),
        )
    if loader is not None:
        raise ValueError(f"Unknown dataset loader: {loader!r}")

    path = spec["path"]
    load_kwargs: dict[str, Any] = {
        "split": spec["split"],
        "streaming": spec["streaming"],
        **kwargs,
    }
    if spec.get("data_dir") is not None:
        load_kwargs["data_dir"] = spec["data_dir"]

    config = spec.get("config")
    if config is not None:
        return load_dataset(path, config, **load_kwargs)
    return load_dataset(path, **load_kwargs)


dataset_registry = {
    "realnewslike": {
        "path": "allenai/c4",
        "config": "realnewslike",
        "split": "validation",
        "data_field": "text",
        "streaming": False,
    },
    "wikipedia": {
        "path": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "data_field": "text",
        "streaming": True,
    },
    "arxiv": {
        "path": "armanc/scientific_papers",
        "config": "arxiv",
        "split": "test",
        "data_field": "article",
        "streaming": False,
    },
    "booksum": {
        "path": "kmfoda/booksum",
        "config": None,
        "split": "test",
        "data_field": "chapter",
        "streaming": False,
    },
    "openwebmath": {
        "path": "open-web-math/open-web-math",
        "config": None,
        "split": "train",
        "data_field": "text",
        "streaming": True,
    },
    "c4_ja": {
        "path": "allenai/c4",
        "config": "ja",
        "split": "validation",
        "data_field": "text",
        "streaming": True,
    },
    "fineweb2_ja": {
        "path": "HuggingFaceFW/fineweb-2",
        "config": "jpn_Jpan",
        "split": "test",
        "data_field": "text",
        "streaming": True,
    },
    "c4_de": {
        "path": "allenai/c4",
        "config": "de",
        "split": "validation",
        "data_field": "text",
        "streaming": True,
    },
    "c4_ta": {
        "path": "allenai/c4",
        "config": "ta",
        "split": "validation",
        "data_field": "text",
        "streaming": True,
    },
    # Gated on HF: accept The Stack terms and run `huggingface-cli login`.
    # Load one language subset via data_dir (see bigcode/starcoderdata README).
    "starcoderdata": {
        "path": "bigcode/starcoderdata",
        "config": None,
        "data_dir": "python",
        "split": "train",
        "data_field": "content",
        "streaming": True,
    },
    # Official AlpacaEval / AlpacaFarm eval instructions (805 prompts).
    # Separate from tatsu-lab/alpaca FT data; instruction-prompted generation.
    # Protocol: gold output ≥ 50 tokens (~574/805), shuffle seed 42, first 500.
    "alpaca_eval": {
        "path": "tatsu-lab/alpaca_eval",
        "config": None,
        "split": "eval",
        "data_file": "alpaca_eval.json",
        "data_field": "prompt",
        "completion_field": "completion",
        "streaming": False,
        "loader": "alpaca_eval",
        "prompt_mode": "instruction",
    },
}
