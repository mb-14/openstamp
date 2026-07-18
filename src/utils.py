from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

large_models = [
    "meta-llama/Llama-3.1-70B"
]

VALID_QUANTIZATIONS = (None, "none", "nf4", "int8")


def _normalize_quantization(quantization):
    if quantization is None or quantization == "none":
        return None
    q = str(quantization).lower().strip()
    if q not in {"nf4", "int8"}:
        raise ValueError(
            f"Unsupported quantization={quantization!r}; "
            f"expected one of none, nf4, int8"
        )
    return q


def _bitsandbytes_config(quantization: str) -> BitsAndBytesConfig:
    if quantization == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization={quantization!r}")


def load_model(model_name: str, quantization=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = _normalize_quantization(quantization)

    # Large models default to INT8 unless an explicit scheme is requested.
    if quantization is None and model_name in large_models:
        print("Quantizing model (default INT8 for 70B)")
        quantization = "int8"

    if quantization is not None:
        print(f"Loading model with bitsandbytes quantization={quantization}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=_bitsandbytes_config(quantization),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer
