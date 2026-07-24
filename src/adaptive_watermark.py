"""Adaptive Text Watermark (Liu & Bu, ICML 2024).

Integrates as a HF LogitsProcessor (same generation path as KGW) plus
score_text_batch detection.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from src.adaptive.semantic_model import SemanticModel

DEFAULT_SMM_PATH = (
    Path(__file__).resolve().parent / "adaptive" / "semantic_mapping_model.pth"
)
DEFAULT_SECRET = "The quick brown fox jumps over the lazy dog"
DEFAULT_EMBEDDER = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_MEASURE_MODEL = "openai-community/gpt2-large"
SMM_OUTPUT_DIM = 384
# Chunk sizes for batched detection (entropy / SBERT).
ENTROPY_CHUNK = 32
EMBED_CHUNK = 64


def vocabulary_mapping(
    vocab_size: int, model_output_dim: int = SMM_OUTPUT_DIM, seed: int = 66
) -> List[int]:
    rng = random.Random(seed)
    return [rng.randint(0, model_output_dim - 1) for _ in range(vocab_size)]


class AdaptiveLogitsProcessor(LogitsProcessor):
    """Bias next-token logits using semantic mapping + entropy gating."""

    def __init__(
        self,
        tokenizer,
        measure_model,
        measure_tokenizer,
        embedding_model: SentenceTransformer,
        transform_model: SemanticModel,
        mapping_list: Sequence[int],
        device: torch.device,
        prompt_length: int,
        alpha: float = 2.0,
        delta: float = 1.5,
        delta_0: float = 1.0,
        measure_threshold: int = 50,
        secret_string: str = DEFAULT_SECRET,
    ):
        self.tokenizer = tokenizer
        self.measure_model = measure_model
        self.measure_tokenizer = measure_tokenizer
        self.embedding_model = embedding_model
        self.transform_model = transform_model
        self.device = device
        self.prompt_length = int(prompt_length)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.delta_0 = float(delta_0)
        self.measure_threshold = int(measure_threshold)
        self.secret_string = secret_string

        if self.measure_tokenizer.pad_token is None:
            self.measure_tokenizer.pad_token = self.measure_tokenizer.eos_token

        self.mapping_index = torch.tensor(
            list(mapping_list), device=device, dtype=torch.long
        )
        self._secret_v_embedding = self._text_to_v_embedding(self.secret_string)

    def _embed_signs_batch(self, texts: List[str]) -> torch.Tensor:
        """Return (B, smm_dim) sign vectors for a list of texts."""
        if not texts:
            return torch.empty(0, SMM_OUTPUT_DIM, device=self.device)
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                texts,
                convert_to_tensor=True,
                device=str(self.device),
                batch_size=min(EMBED_CHUNK, len(texts)),
                show_progress_bar=False,
            )
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            transformed = self.transform_model(embedding.float())
            signs = (transformed > 0).float()
        return signs

    def _embed_signs(self, text: str) -> torch.Tensor:
        return self._embed_signs_batch([text])[0]

    def _text_to_v_embedding(self, text: str) -> torch.Tensor:
        signs = self._embed_signs(text)
        return signs[self.mapping_index]

    def _batch_text_to_v_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        if not texts:
            return []
        signs = self._embed_signs_batch(texts)
        return [signs[i][self.mapping_index] for i in range(signs.size(0))]

    def _next_token_entropy(self, text: str) -> float:
        return self._batch_next_token_entropy([text])[0]

    def _batch_next_token_entropy(self, texts: List[str]) -> List[float]:
        """Batched next-token entropy under the measure model."""
        if not texts:
            return []

        out: List[float] = [0.0] * len(texts)
        nonempty = [(i, t) for i, t in enumerate(texts) if t and str(t).strip()]
        if not nonempty:
            return out

        for start in range(0, len(nonempty), ENTROPY_CHUNK):
            chunk = nonempty[start : start + ENTROPY_CHUNK]
            chunk_texts = [t for _, t in chunk]
            enc = self.measure_tokenizer(
                chunk_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                add_special_tokens=False,
            ).to(self.device)
            with torch.no_grad():
                logits = self.measure_model(**enc).logits  # (B, T, V)
                attn = enc["attention_mask"]
                last_idx = attn.sum(dim=1) - 1
                rows = torch.arange(logits.size(0), device=logits.device)
                last_logits = logits[rows, last_idx].float()
                probs = torch.softmax(last_logits, dim=-1)
                ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
            for (orig_i, _), e in zip(chunk, ent.tolist()):
                out[orig_i] = float(e)
        return out

    def _bias_logits(
        self, logits: torch.Tensor, v_embedding: torch.Tensor, delta: float
    ) -> torch.Tensor:
        v = v_embedding
        if v.numel() != logits.numel():
            if v.numel() > logits.numel():
                v = v[: logits.numel()]
            else:
                pad = torch.zeros(
                    logits.numel() - v.numel(), device=logits.device, dtype=v.dtype
                )
                v = torch.cat([v, pad], dim=0)
        return logits * (1.0 + delta * v)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        gen_len = int(input_ids.shape[1]) - self.prompt_length
        if gen_len < 0:
            gen_len = int(input_ids.shape[1])

        for b in range(batch_size):
            if gen_len <= self.measure_threshold:
                scores[b] = self._bias_logits(
                    scores[b], self._secret_v_embedding, self.delta_0
                )
                continue

            generated_ids = input_ids[b, self.prompt_length :]
            measure_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            entropy = self._next_token_entropy(measure_text)
            if entropy >= self.alpha:
                v_embedding = self._text_to_v_embedding(measure_text)
                scores[b] = self._bias_logits(scores[b], v_embedding, self.delta)

        return scores


class AdaptiveMark:
    def __init__(
        self,
        tokenizer,
        model=None,
        device: Optional[torch.device] = None,
        prompt_length: int = 50,
        alpha: float = 2.0,
        delta: float = 1.5,
        delta_0: float = 1.0,
        measure_threshold: int = 50,
        secret_string: str = DEFAULT_SECRET,
        measure_model_name: str = DEFAULT_MEASURE_MODEL,
        embedder_name: str = DEFAULT_EMBEDDER,
        smm_path: Optional[str] = None,
        mapping_seed: int = 66,
        measure_model=None,
        measure_tokenizer=None,
        embedding_model: Optional[SentenceTransformer] = None,
        transform_model: Optional[SemanticModel] = None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        if device is None:
            if model is not None and hasattr(model, "device"):
                device = model.device
            else:
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        self.device = torch.device(device)

        self.alpha = float(alpha)
        self.delta = float(delta)
        self.delta_0 = float(delta_0)
        self.measure_threshold = int(measure_threshold)
        self.secret_string = secret_string
        self.measure_model_name = measure_model_name
        self.embedder_name = embedder_name
        self.smm_path = str(smm_path or DEFAULT_SMM_PATH)
        self.mapping_seed = int(mapping_seed)
        self.prompt_length = int(prompt_length)

        if measure_model is None or measure_tokenizer is None:
            measure_tokenizer = AutoTokenizer.from_pretrained(measure_model_name)
            measure_model = AutoModelForCausalLM.from_pretrained(
                measure_model_name,
                torch_dtype=(
                    torch.float16 if self.device.type == "cuda" else torch.float32
                ),
            ).to(self.device)
            measure_model.eval()
        self.measure_model = measure_model
        self.measure_tokenizer = measure_tokenizer

        if embedding_model is None:
            embedding_model = SentenceTransformer(
                embedder_name, device=str(self.device)
            )
            embedding_model.eval()
        self.embedding_model = embedding_model

        if transform_model is None:
            transform_model = SemanticModel()
            state = torch.load(self.smm_path, map_location=self.device, weights_only=True)
            transform_model.load_state_dict(state)
            transform_model.to(self.device)
            transform_model.eval()
        self.transform_model = transform_model

        vocab_size = len(tokenizer)
        self.mapping_list = vocabulary_mapping(
            vocab_size, SMM_OUTPUT_DIM, seed=self.mapping_seed
        )

        self.watermark = AdaptiveLogitsProcessor(
            tokenizer=tokenizer,
            measure_model=self.measure_model,
            measure_tokenizer=self.measure_tokenizer,
            embedding_model=self.embedding_model,
            transform_model=self.transform_model,
            mapping_list=self.mapping_list,
            device=self.device,
            prompt_length=self.prompt_length,
            alpha=self.alpha,
            delta=self.delta,
            delta_0=self.delta_0,
            measure_threshold=self.measure_threshold,
            secret_string=self.secret_string,
        )

    def _score_one(self, text: str) -> float:
        """Mean green-list score over tokens (official detection())."""
        return float(self.score_text_batch([text])[0].item())

    def score_text_batch(self, batch_text: List[str]) -> torch.Tensor:
        """Batched detection: shared measure/SBERT forwards across prefixes."""
        proc = self.watermark
        secret_ve = proc._secret_v_embedding
        results = [0.0] * len(batch_text)

        # Per-text token ids and early (secret) scores.
        all_ids: List[Optional[torch.Tensor]] = []
        all_scores: List[List[float]] = []
        # Flat list of late-token work: (text_idx, token_pos, prefix_str)
        late_jobs: List[tuple] = []

        for ti, text in enumerate(batch_text):
            if not text or not str(text).strip():
                all_ids.append(None)
                all_scores.append([])
                continue

            ids = self.tokenizer.encode(
                text, return_tensors="pt", add_special_tokens=False
            )[0]
            all_ids.append(ids)
            scores: List[float] = []
            n = int(ids.numel())
            early_end = min(n, self.measure_threshold + 1)
            for i in range(early_end):
                tok = int(ids[i].item())
                s = float(secret_ve[tok].item()) if tok < secret_ve.numel() else 0.0
                scores.append(s)
            all_scores.append(scores)

            for i in range(self.measure_threshold + 1, n):
                prefix = self.tokenizer.decode(ids[:i], skip_special_tokens=True)
                late_jobs.append((ti, i, prefix))

        if late_jobs:
            prefixes = [p for _, _, p in late_jobs]
            entropies = proc._batch_next_token_entropy(prefixes)
            embed_jobs = [
                (j, late_jobs[j][0], late_jobs[j][1], late_jobs[j][2])
                for j, e in enumerate(entropies)
                if e >= self.alpha
            ]
            if embed_jobs:
                ve_list = proc._batch_text_to_v_embeddings(
                    [p for _, _, _, p in embed_jobs]
                )
                for k, (_, ti, pos, _) in enumerate(embed_jobs):
                    ids = all_ids[ti]
                    assert ids is not None
                    tok = int(ids[pos].item())
                    ve = ve_list[k]
                    s = float(ve[tok].item()) if tok < ve.numel() else 0.0
                    all_scores[ti].append(s)

        for ti, scores in enumerate(all_scores):
            if scores:
                results[ti] = float(sum(scores) / len(scores))

        return torch.tensor(results, dtype=torch.float32)
