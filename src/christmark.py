"""Christ et al. (arXiv:2410.18861) open-source watermark baseline.

Embeds a secret Gaussian key Δ ~ N(0, ε²I) into the final-layer bias
(lm_head.bias). Detection scores text by the mean of Δ over distinct token IDs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChristMark:
    def __init__(self, epsilon, seed, tokenizer, model=None, vocab_size=None):
        """
        Args:
            epsilon: stddev of Gaussian bias key (ε in the paper).
            seed: seed for deterministic watermark key.
            tokenizer: HuggingFace tokenizer.
            model: causal LM to watermark (required for generation; None for detect-only).
            vocab_size: override vocab size for the key; defaults to model config or tokenizer.
        """
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.tokenizer = tokenizer

        if vocab_size is None:
            if model is not None and getattr(model.config, "vocab_size", None):
                vocab_size = int(model.config.vocab_size)
            else:
                vocab_size = len(tokenizer)
        self.vocab_size = int(vocab_size)
        self.delta = self._watermark_key(self.vocab_size)

        self.model = None
        if model is not None:
            self.watermark_model(model)

    def _watermark_key(self, vocab_size: int) -> torch.Tensor:
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        return torch.randn(vocab_size, generator=rng) * self.epsilon

    def watermark_model(self, model):
        """Add Δ to lm_head.bias, creating a zero bias if the layer has none."""
        lm_head = model.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("Model has no output embeddings / lm_head.")

        out_features = lm_head.weight.shape[0]
        if out_features != self.vocab_size:
            # Rebuild key to match the actual unembedding rows.
            self.vocab_size = out_features
            self.delta = self._watermark_key(self.vocab_size)

        with torch.no_grad():
            if lm_head.bias is None:
                lm_head.bias = nn.Parameter(
                    torch.zeros(
                        self.vocab_size,
                        device=lm_head.weight.device,
                        dtype=lm_head.weight.dtype,
                    )
                )

            key = self.delta.to(device=lm_head.bias.device, dtype=lm_head.bias.dtype)
            lm_head.bias.data.add_(key)

        self.model = model

    def score_text_batch(self, batch_text):
        """Return mean Δ over distinct token IDs for each text (higher = more watermarked)."""
        inputs = self.tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        special_ids = set(self.tokenizer.all_special_ids or [])
        scores = []
        delta = self.delta

        for i in range(input_ids.shape[0]):
            ids = input_ids[i][attention_mask[i].bool()].tolist()
            unique = []
            seen = set()
            for tok in ids:
                if tok in special_ids:
                    continue
                if tok < 0 or tok >= self.vocab_size:
                    continue
                if tok not in seen:
                    seen.add(tok)
                    unique.append(tok)
            if not unique:
                scores.append(0.0)
            else:
                scores.append(delta[unique].mean().item())

        return torch.tensor(scores, dtype=torch.float32)
