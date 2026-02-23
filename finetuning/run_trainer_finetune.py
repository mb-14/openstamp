import torch
torch.set_float32_matmul_precision("high")

from itertools import chain
from rich import print as pprint
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import ModelConfig, SFTConfig, TrlParser, SFTTrainer
from peft import LoraConfig, get_peft_model
from src.mbmark import MbMark, Mode
from src.gaussmark import GaussMark
from dataclasses import dataclass
import os


@dataclass
class CustomArgs:
    selector_matrix_dir: str
    watermark_seed: int
    watermark_type: str = "openstamp"



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


def tokenize_dataset(dataset, tokenizer, sequence_length: int = 200):
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns="text",
    )
    lm_dataset = tokenized_dataset.map(
        lambda examples: group_texts(examples, sequence_length),
        batched=True,
    )

    return lm_dataset

def main():
    parser = TrlParser((SFTConfig, ModelConfig, CustomArgs))
    training_args, models_args, custom_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    print(custom_args)

    #! load the model
    model_kwargs = {
        "attn_implementation": models_args.attn_implementation,
        "torch_dtype": "bfloat16",
        "use_cache": False if training_args.gradient_checkpointing else True,
    }

    model = AutoModelForCausalLM.from_pretrained(models_args.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    watermark_type = custom_args.watermark_type

    if watermark_type == "openstamp":
        selector_matrix_dir = custom_args.selector_matrix_dir
        selector_matrix_path = os.path.join(selector_matrix_dir, "selector_matrix.pth")
        final_weight = torch.load(selector_matrix_path)
        mb_mark = MbMark.mb(
            delta=1.0,
            gamma=0.25,
            seed=custom_args.watermark_seed,
            final_weight=final_weight,
            model=model,
            unembedding_param_name="lm_head",
            tokenizer=tokenizer,
            mode=Mode.Generate,
        )

        model = mb_mark.model
    if watermark_type == "gaussmark":
        target_param_name = "model.layers.27.mlp.up_proj.weight"
        sigma = 0.04
        gaussmark = GaussMark(sigma=sigma, seed=custom_args.watermark_seed,
                          target_param_name=target_param_name, tokenizer=tokenizer, model=model)
        model = gaussmark.model

    
    #! use the fineweb-edu dataset
    # dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=False)

    dataset = load_dataset("Skylion007/openwebtext", split="train[0:500000]", num_proc=32, trust_remote_code=True, streaming=False)
    dataset = dataset.shuffle(seed=42).select(range(500000))
    column_names = dataset.column_names
    dataset = dataset.remove_columns([col for col in column_names if col != "text"])
    dataset = tokenize_dataset(dataset, tokenizer, sequence_length=min(512, tokenizer.model_max_length))

    #! lora config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        fan_in_fan_out=False,
        bias="none",
        target_modules=["v_proj", "k_proj", "o_proj", "q_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # for param in model.parameters():
    #     param.requires_grad = False
        
    # # Unfreeze lm_head parameters
    # for param in model.lm_head.parameters():
    #     param.requires_grad = True

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
    