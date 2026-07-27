"""Alpaca helpers for instruction SFT and AlpacaEval prompting.

Canonical instruction FT uses the full ``tatsu-lab/alpaca`` set (~52k rows)
formatted as prompt/completion for TRL response-only loss.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

ALPACA_DATASET_NAME = "tatsu-lab/alpaca"
ALPACA_EVAL_DATASET_NAME = "tatsu-lab/alpaca_eval"
ALPACA_EVAL_DATA_FILE = "alpaca_eval.json"
# Generation-eval protocol: keep rows whose gold output has at least this many tokens.
ALPACA_EVAL_MIN_GOLD_TOKENS = 50

ALPACA_PROMPT_PREFIX = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
)


def format_alpaca_prompt(instruction: str, input_text: str | None = None) -> str:
    """Build the Alpaca prompt up to and including the Response header."""
    instruction = instruction or ""
    input_text = (input_text or "").strip()
    if input_text:
        return (
            f"{ALPACA_PROMPT_PREFIX}"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    return (
        f"{ALPACA_PROMPT_PREFIX}"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n"
    )


def format_alpaca_sft_row(example: dict) -> dict[str, str]:
    """Format one Alpaca row for TRL prompt-completion SFT (response-only loss)."""
    return {
        "prompt": format_alpaca_prompt(example.get("instruction"), example.get("input")),
        "completion": example.get("output") or "",
    }


def load_alpaca_sft_dataset(*, num_proc: int | None = 8) -> Dataset:
    """Load full ``tatsu-lab/alpaca`` as prompt/completion columns for SFTTrainer."""
    dataset = load_dataset(ALPACA_DATASET_NAME, split="train", num_proc=num_proc)
    dataset = dataset.map(format_alpaca_sft_row, num_proc=num_proc)
    keep = ["prompt", "completion"]
    drop = [c for c in dataset.column_names if c not in keep]
    if drop:
        dataset = dataset.remove_columns(drop)
    print(
        f"Alpaca SFT: {len(dataset)} examples "
        f"(full {ALPACA_DATASET_NAME}, prompt/completion, response-only loss)"
    )
    return dataset


def format_alpaca_eval_row(example: dict) -> dict[str, str]:
    """Format one AlpacaEval row as prompt/completion for generation evals."""
    return {
        "prompt": format_alpaca_prompt(example["instruction"], ""),
        "completion": example.get("output") or "",
    }


def load_alpaca_eval_dataset(
    *,
    repo_id: str = ALPACA_EVAL_DATASET_NAME,
    data_file: str = ALPACA_EVAL_DATA_FILE,
) -> Dataset:
    """Load official AlpacaEval instructions (805 prompts) as prompt/completion.

    Downloads the raw JSON via the Hub (dataset scripts are unsupported in
    recent ``datasets``) and formats each row with the Alpaca prompt template.
    """
    json_path = hf_hub_download(
        repo_id=repo_id,
        filename=data_file,
        repo_type="dataset",
    )
    return Dataset.from_json(json_path).map(format_alpaca_eval_row)


def alpaca_eval_gold_ok(
    example: dict,
    tokenizer,
    *,
    completion_field: str = "completion",
    min_gold_tokens: int = ALPACA_EVAL_MIN_GOLD_TOKENS,
) -> bool:
    """True if the gold completion is non-empty and ≥ ``min_gold_tokens`` long."""
    completion = example.get(completion_field) or ""
    if not str(completion).strip():
        return False
    n_tok = len(
        tokenizer(str(completion), truncation=False, add_special_tokens=False)[
            "input_ids"
        ]
    )
    return n_tok >= min_gold_tokens


def filter_alpaca_eval_dataset(
    dataset: Dataset,
    tokenizer,
    *,
    completion_field: str = "completion",
    min_gold_tokens: int = ALPACA_EVAL_MIN_GOLD_TOKENS,
) -> Dataset:
    """Apply the AlpacaEval gold-length filter used for generation evals."""
    return dataset.filter(
        lambda x: alpaca_eval_gold_ok(
            x,
            tokenizer,
            completion_field=completion_field,
            min_gold_tokens=min_gold_tokens,
        )
    )


def instruction_prompt_ok(
    example: dict,
    tokenizer,
    *,
    prompt_field: str = "prompt",
    completion_field: str = "completion",
    prompt_length: int,
    min_prompt_tokens: int = 8,
) -> bool:
    """Keep instruction rows with a usable prompt + non-empty gold output."""
    prompt = example[prompt_field]
    completion = example.get(completion_field) or ""
    if not str(completion).strip():
        return False
    n_tok = len(tokenizer(prompt, truncation=False)["input_ids"])
    return min_prompt_tokens <= n_tok <= prompt_length


def filter_instruction_prompts(
    dataset: Dataset,
    tokenizer,
    *,
    prompt_field: str = "prompt",
    completion_field: str = "completion",
    prompt_length: int,
    min_prompt_tokens: int = 8,
) -> Dataset:
    """Filter instruction rows that fit the fixed generation prompt budget."""
    return dataset.filter(
        lambda x: instruction_prompt_ok(
            x,
            tokenizer,
            prompt_field=prompt_field,
            completion_field=completion_field,
            prompt_length=prompt_length,
            min_prompt_tokens=min_prompt_tokens,
        )
    )


def encode_instruction(
    example: dict,
    tokenizer,
    *,
    prompt_field: str = "prompt",
    completion_field: str = "completion",
    prompt_length: int,
    device=None,
) -> dict:
    """Encode an Alpaca-style instruction prompt (left-pad to ``prompt_length``)."""
    prompt_text_raw = example[prompt_field]
    completion = example[completion_field] or ""
    full_text = f"{prompt_text_raw}{completion}"

    # Left-pad so batch generation aligns on the rightmost real tokens.
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        prompt = tokenizer(
            prompt_text_raw,
            truncation=True,
            padding="max_length",
            max_length=prompt_length,
            return_tensors="pt",
        )
        if device is not None:
            prompt = prompt.to(device)
    finally:
        tokenizer.padding_side = prev_side

    prompt_text = tokenizer.batch_decode(
        prompt["input_ids"], skip_special_tokens=True
    )[0]

    return {
        "text": full_text,
        "prompt_text": prompt_text,
        "chat_prompt_text": prompt_text,
        "input_ids": prompt["input_ids"].squeeze(0),
        "attention_mask": prompt["attention_mask"].squeeze(0),
        "text_completion": completion,
    }
