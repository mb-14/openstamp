#!/usr/bin/env python3
"""Write deterministic SIR vocab mappings for Llama-2-7B and Mistral-7B."""

from src.sir import ensure_mapping

SEED = 15485863
for vocab_size in (32000, 32768):
    path = ensure_mapping(vocab_size, SEED)
    print(path)
