import torch
torch.set_float32_matmul_precision("high")

from itertools import chain
from rich import print as pprint
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import ModelConfig, SFTConfig, TrlParser, SFTTrainer
from peft import LoraConfig, get_peft_model
from src.openstamp import OpenStamp, Mode
from src.gaussmark import GaussMark
from src.christmark import ChristMark
from dataclasses import dataclass
import os


# Fixed method hyperparameters (same as main eval configs).
OPENSTAMP_DELTA = 1.0
OPENSTAMP_GAMMA = 0.25
OPENSTAMP_UNEMBEDDING_PARAM_NAME = "lm_head"

DATASET_NAME = "Skylion007/openwebtext"
DATASET_N_DOCS = 500_000
DATASET_SHUFFLE_SEED = 739
DATASET_CACHE_ROOT = os.path.join("finetuning", "cache")

GAUSSMARK_CONFIG = {
    "meta-llama/Llama-2-7b-hf": {
        "target_param_name": "model.layers.27.mlp.up_proj.weight",
        "sigma": 0.04,
    },
    "mistralai/Mistral-7B-v0.3": {
        "target_param_name": "model.layers.20.mlp.up_proj.weight",
        "sigma": 0.005,
    },
}

CHRIST_EPSILON = 0.8


@dataclass
class CustomArgs:
    selector_matrix_dir: str
    watermark_seed: int
    watermark_type: str = "openstamp"
    target_param_config: int = 0


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples, sequence_length):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // sequence_length) * sequence_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_dataset(dataset, tokenizer, sequence_length: int = 200, num_proc: int = 32):
    tokenized_dataset = dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns="text",
        num_proc=num_proc,
        desc="Tokenizing",
    )
    lm_dataset = tokenized_dataset.map(
        group_texts,
        fn_kwargs={"sequence_length": sequence_length},
        batched=True,
        num_proc=num_proc,
        desc="Grouping texts",
    )
    return lm_dataset


def lm_dataset_cache_dir(model_name_or_path: str, sequence_length: int) -> str:
    model_slug = model_name_or_path.replace("/", "__")
    return os.path.join(
        DATASET_CACHE_ROOT,
        f"openwebtext_{model_slug}_n{DATASET_N_DOCS}_seed{DATASET_SHUFFLE_SEED}_seq{sequence_length}",
    )


def load_or_build_lm_dataset(tokenizer, model_name_or_path: str, sequence_length: int, num_proc: int = 8):
    cache_dir = lm_dataset_cache_dir(model_name_or_path, sequence_length)
    if os.path.isdir(cache_dir):
        print(f"Loading cached LM dataset from {cache_dir}")
        return load_from_disk(cache_dir)

    print(f"Building LM dataset cache at {cache_dir}")
    dataset = load_dataset(
        DATASET_NAME,
        split=f"train[0:{DATASET_N_DOCS}]",
        num_proc=num_proc,
        trust_remote_code=True,
        streaming=False,
    )
    dataset = dataset.shuffle(seed=DATASET_SHUFFLE_SEED).select(range(DATASET_N_DOCS))
    column_names = dataset.column_names
    dataset = dataset.remove_columns([col for col in column_names if col != "text"])
    dataset = tokenize_dataset(
        dataset, tokenizer, sequence_length=sequence_length, num_proc=num_proc
    )
    os.makedirs(DATASET_CACHE_ROOT, exist_ok=True)
    dataset.save_to_disk(cache_dir)
    print(f"Saved LM dataset cache to {cache_dir}")
    return dataset


def main():
    parser = TrlParser((SFTConfig, ModelConfig, CustomArgs))
    training_args, models_args, custom_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    print(custom_args)

    tokenizer = AutoTokenizer.from_pretrained(models_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sequence_length = min(512, tokenizer.model_max_length)
    num_proc = getattr(training_args, "dataset_num_proc", None) or 32
    dataset = load_or_build_lm_dataset(
        tokenizer,
        models_args.model_name_or_path,
        sequence_length=sequence_length,
        num_proc=num_proc,
    )

    #! load the model
    model_kwargs = {
        "attn_implementation": models_args.attn_implementation,
        "torch_dtype": "bfloat16",
        "use_cache": False if training_args.gradient_checkpointing else True,
    }

    model = AutoModelForCausalLM.from_pretrained(models_args.model_name_or_path, **model_kwargs)

    # FlashAttention (and ChristMark's bias check) need the model on CUDA before any forward.
    if torch.cuda.is_available():
        model = model.cuda()

    watermark_type = custom_args.watermark_type

    if watermark_type == "openstamp":
        selector_matrix_dir = custom_args.selector_matrix_dir
        selector_matrix_path = os.path.join(selector_matrix_dir, "selector_matrix.pth")
        final_weight = torch.load(selector_matrix_path)
        mb_mark = OpenStamp.from_config(
            delta=OPENSTAMP_DELTA,
            gamma=OPENSTAMP_GAMMA,
            seed=custom_args.watermark_seed,
            final_weight=final_weight,
            model=model,
            unembedding_param_name=OPENSTAMP_UNEMBEDDING_PARAM_NAME,
            tokenizer=tokenizer,
            mode=Mode.Generate,
        )
        model = mb_mark.model
    elif watermark_type == "gaussmark":
        gm_cfg = GAUSSMARK_CONFIG.get(models_args.model_name_or_path)
        if gm_cfg is None:
            raise ValueError(
                f"No GaussMark config for model {models_args.model_name_or_path}. "
                f"Supported: {sorted(GAUSSMARK_CONFIG)}"
            )
        gaussmark = GaussMark(
            sigma=gm_cfg["sigma"],
            seed=custom_args.watermark_seed,
            target_param_name=gm_cfg["target_param_name"],
            tokenizer=tokenizer,
            model=model,
        )
        model = gaussmark.model
    elif watermark_type in {"unremovable", "christ"}:
        christ = ChristMark(
            epsilon=CHRIST_EPSILON,
            seed=custom_args.watermark_seed,
            tokenizer=tokenizer,
            model=model,
        )
        model = christ.model
    elif watermark_type == "kgw_distilled":
        # Distilled checkpoint is already watermarked; no embedding step.
        pass
    else:
        raise ValueError(f"Unsupported watermark_type: {watermark_type}")

    if custom_args.target_param_config == 2:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze lm_head parameters
        for param in model.lm_head.parameters():
            param.requires_grad = True
        
        lora_config = None

    else:    
        target_modules = ["v_proj", "k_proj", "o_proj", "q_proj", "gate_proj", "up_proj", "down_proj"]
        #! lora config
        if custom_args.target_param_config == 1:
            target_modules.append("lm_head")

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            fan_in_fan_out=False,
            bias="none",
            target_modules=target_modules
        )


    #! train the model
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
        peft_config=lora_config,
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
    