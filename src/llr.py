"""Length-normalized log-likelihood ratio used by OpenStamp-style detectors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def length_normalized_llr(
    logits_base: torch.Tensor,
    logits_marked: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence LLR of marked vs base next-token distributions.

    Uses the OpenStamp / Unremovable / KGW+LLR shift: skip the first token
    (typically BOS), predict ``input_ids[:, 2:]`` from logits at ``[:, 1:-1]``,
    and divide by the number of unpadded scored tokens.

    Args:
        logits_base: ``(B, T, V)`` unwatermarked logits.
        logits_marked: ``(B, T, V)`` watermarked logits.
        input_ids: ``(B, T)`` token ids.
        attention_mask: ``(B, T)`` padding mask.

    Returns:
        ``(B,)`` float32 CPU tensor; higher means more watermarked.
    """
    log_probs_base = F.log_softmax(logits_base.float(), dim=-1)
    log_probs_marked = F.log_softmax(logits_marked.float(), dim=-1)

    labels = input_ids[:, 2:]
    log_probs_base = log_probs_base[:, 1:-1, :]
    log_probs_marked = log_probs_marked[:, 1:-1, :]
    token_mask = attention_mask[:, 2:].bool()

    log_probs_base = log_probs_base.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    log_probs_marked = log_probs_marked.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    log_probs_base = log_probs_base.masked_fill(~token_mask, 0.0)
    log_probs_marked = log_probs_marked.masked_fill(~token_mask, 0.0)

    lengths = token_mask.sum(dim=1).clamp(min=1).float()
    llr = (log_probs_marked.sum(dim=1) - log_probs_base.sum(dim=1)) / lengths
    return llr.cpu().float()
