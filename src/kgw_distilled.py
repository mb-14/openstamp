from typing import Dict, Tuple

from src.kgw.watermark_processor import WatermarkDetector
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


class KGWDistilled:
    def __init__(self, delta, gamma, seeding_scheme, hash_key, kgw_device, tokenizer, model=None):
        self.tokenizer = tokenizer
        if model is not None:
            self.model = model
            self.model.eval()
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
        all_scores = []
        for text in batch_text:
            score = self.detector.detect(text)
            z_score = score["z_score"]
            all_scores.append(z_score)
        all_scores = torch.tensor(all_scores, dtype=torch.float32)
        return all_scores
