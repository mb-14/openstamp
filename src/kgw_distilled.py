from typing import Dict, Tuple

from src.kgw.watermark_processor import WatermarkDetector
from src.llr import length_normalized_llr
import torch

# Base model + watermark seed (hash key) -> distilled HF checkpoint.
# Seed 15485863 for Llama uses the original cygu release; other seeds are
# mbakshi1094 re-trains with the hash key embedded in the repo name.
DISTILLED_MODEL_BY_SEED: Dict[Tuple[str, int], str] = {
    ("meta-llama/Llama-2-7b-hf", 12997009): (
        "mbakshi1094/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2-hk12997009-legacy"
    ),
    ("meta-llama/Llama-2-7b-hf", 22983996): (
        "mbakshi1094/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2-hk22983996-legacy"
    ),
    ("meta-llama/Llama-2-7b-hf", 15485863): (
        "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2"
    ),
    ("mistralai/Mistral-7B-v0.3", 12997009): (
        "mbakshi1094/mistral-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2-hk12997009-legacy"
    ),
    ("mistralai/Mistral-7B-v0.3", 22983996): (
        "mbakshi1094/mistral-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2-hk22983996-legacy"
    ),
    ("mistralai/Mistral-7B-v0.3", 15485863): (
        "mbakshi1094/mistral-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2-hk15485863-legacy"
    ),
}


def resolve_distilled_model(base_model: str, seed: int) -> str:
    """Map a base LM + watermark seed to the matching KGW-distilled HF model ID."""
    key = (base_model, int(seed))
    try:
        return DISTILLED_MODEL_BY_SEED[key]
    except KeyError as exc:
        raise KeyError(
            f"No distilled model for base_model={base_model!r} seed={seed}. "
            f"Known keys: {sorted(DISTILLED_MODEL_BY_SEED)}"
        ) from exc


def resolve_base_model(distilled_id: str) -> str:
    """Map a distilled HF model ID back to its unwatermarked base LM."""
    matches = {
        base for (base, _seed), distilled in DISTILLED_MODEL_BY_SEED.items()
        if distilled == distilled_id
    }
    if len(matches) == 1:
        return next(iter(matches))
    if not matches:
        raise KeyError(
            f"No base model for distilled_id={distilled_id!r}. "
            f"Known distilled IDs: {sorted(set(DISTILLED_MODEL_BY_SEED.values()))}"
        )
    raise KeyError(
        f"Ambiguous base model for distilled_id={distilled_id!r}: {sorted(matches)}"
    )


class KGWDistilled:
    def __init__(
        self,
        delta,
        gamma,
        seeding_scheme,
        hash_key,
        kgw_device,
        tokenizer,
        model=None,
        base_model=None,
        llr_detection=False,
    ):
        self.tokenizer = tokenizer
        self.llr_detection = bool(llr_detection)
        self.model = None
        self.base_model = None
        if model is not None:
            self.model = model
            self.model.eval()
        if base_model is not None:
            self.base_model = base_model
            self.base_model.eval()
        self.detector = WatermarkDetector(
            device=kgw_device,
            tokenizer=tokenizer,
            vocab=tokenizer.get_vocab().values(),
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            hash_key=hash_key,
            normalizers=[],
        )

    def score_text_batch(self, batch_text):
        if self.llr_detection:
            return self.llr_detect(batch_text)
        all_scores = []
        for text in batch_text:
            score = self.detector.detect(text)
            z_score = score["z_score"]
            all_scores.append(z_score)
        all_scores = torch.tensor(all_scores, dtype=torch.float32)
        return all_scores

    @torch.no_grad()
    def llr_detect(self, texts):
        """Length-normalized LLR of distilled vs original next-token dists."""
        if self.model is None or self.base_model is None:
            raise RuntimeError(
                "LLR detection requires both the distilled model and the base model"
            )

        device = next(self.model.parameters()).device
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        logits_marked = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        base_device = next(self.base_model.parameters()).device
        if base_device != device:
            encodings_base = encodings.to(base_device)
            logits_base = self.base_model(
                input_ids=encodings_base.input_ids,
                attention_mask=encodings_base.attention_mask,
            ).logits
            logits_base = logits_base.to(logits_marked.device)
        else:
            logits_base = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        return length_normalized_llr(logits_base, logits_marked, input_ids, attention_mask)
