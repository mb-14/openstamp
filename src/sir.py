"""SIR (Semantic Invariant Robust) watermark, wrapping MarkLLM's implementation."""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKLLM_ROOT = REPO_ROOT / "third_party" / "MarkLLM"
DEFAULT_TRANSFORM = MARKLLM_ROOT / "watermark" / "sir" / "model" / "transform_model_cbert.pth"
DEFAULT_EMBEDDER_HF = "perceptiveshawty/compositional-bert-large-uncased"
DEFAULT_EMBEDDER_LOCAL = MARKLLM_ROOT / "watermark" / "sir" / "model" / "compositional-bert-large-uncased"


def _default_embedder() -> str:
    weight_names = (
        "pytorch_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
    )
    if DEFAULT_EMBEDDER_LOCAL.is_dir() and any(
        (DEFAULT_EMBEDDER_LOCAL / name).is_file() for name in weight_names
    ):
        return str(DEFAULT_EMBEDDER_LOCAL)
    return DEFAULT_EMBEDDER_HF


DEFAULT_EMBEDDER = _default_embedder()
DEFAULT_MAPPING_DIR = MARKLLM_ROOT / "watermark" / "sir" / "mapping"
SCALE_DIMENSION = 300
TRANSFORM_INPUT_DIM = 1024


def _ensure_markllm_on_path() -> None:
    root = str(MARKLLM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def mapping_path_for(vocab_size: int, seed: int, mapping_dir: Optional[str] = None) -> Path:
    directory = Path(mapping_dir) if mapping_dir else DEFAULT_MAPPING_DIR
    return directory / f"300_mapping_{vocab_size}_seed={int(seed)}.json"


def ensure_mapping(
    vocab_size: int,
    seed: int,
    scale_dimension: int = SCALE_DIMENSION,
    path: Optional[str] = None,
) -> str:
    """Write a deterministic vocab→scale-dim map if missing; return its absolute path."""
    dest = Path(path) if path else mapping_path_for(vocab_size, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        with dest.open("r", encoding="utf-8") as handle:
            mapping = json.load(handle)
        if len(mapping) != vocab_size:
            raise ValueError(
                f"SIR mapping {dest} has length {len(mapping)}, expected vocab_size={vocab_size}"
            )
        return str(dest.resolve())

    rng = random.Random(int(seed))
    mapping = [rng.randint(0, scale_dimension - 1) for _ in range(vocab_size)]
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle)
    return str(dest.resolve())


class SIRMark:
    def __init__(
        self,
        tokenizer,
        delta: float = 1.0,
        chunk_length: int = 10,
        seed: int = 15485863,
        model=None,
        vocab_size: Optional[int] = None,
        mapping_name: Optional[str] = None,
        transform_model_name: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.delta = float(delta)
        self.chunk_length = int(chunk_length)
        self.seed = int(seed)
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)
        else:
            self.vocab_size = len(tokenizer)
            if model is not None and hasattr(model, "config"):
                self.vocab_size = max(
                    self.vocab_size,
                    int(getattr(model.config, "vocab_size", self.vocab_size)),
                )
        self.transform_model_name = str(
            Path(transform_model_name).resolve()
            if transform_model_name
            else DEFAULT_TRANSFORM.resolve()
        )
        if not Path(self.transform_model_name).is_file():
            raise FileNotFoundError(
                f"SIR transform weights not found at {self.transform_model_name}. "
                "Download Generative-Watermark-Toolkits/MarkLLM-sir "
                "transform_model_cbert.pth into third_party/MarkLLM/watermark/sir/model/"
            )
        self.embedding_model_path = embedding_model_path or _default_embedder()
        self.mapping_name = ensure_mapping(
            vocab_size=self.vocab_size,
            seed=self.seed,
            path=mapping_name,
        )
        if device is not None:
            self.device = device
        elif model is not None:
            self.device = str(next(model.parameters()).device)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _ensure_markllm_on_path()
        from utils.transformers_config import TransformersConfig
        from watermark.sir.sir import SIR

        transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        config_payload = {
            "algorithm_name": "SIR",
            "delta": self.delta,
            "chunk_length": self.chunk_length,
            "scale_dimension": SCALE_DIMENSION,
            "z_threshold": 0.2,
            "transform_model_input_dim": TRANSFORM_INPUT_DIM,
            "transform_model_name": self.transform_model_name,
            "embedding_model_path": self.embedding_model_path,
            "mapping_name": self.mapping_name,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(config_payload, handle)
            config_path = handle.name

        _orig_load = torch.load

        def _torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _torch_load
        try:
            self._sir = SIR(config_path, transformers_config)
        finally:
            torch.load = _orig_load
        self.watermark = self._sir.logits_processor

    def score_text_batch(self, batch_text):
        scores = []
        for text in batch_text:
            result = self._sir.detect_watermark(text, return_dict=True)
            score = result["score"]
            if score is None or (isinstance(score, float) and math.isnan(score)):
                score = 0.0
            scores.append(float(score))
        return torch.tensor(scores, dtype=torch.float32)
