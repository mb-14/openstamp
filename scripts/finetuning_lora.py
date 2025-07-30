import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.mbmark import MbMark, Mode
from src.gaussmark import GaussMark
from src.kgw_distilled import KGWDistilled
import json
import timeit
import argparse


torch.manual_seed(42)


# Get model name from command line argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="meta-llama/Llama-2-7b-hf",
    help="Name of the model to be used",
)

parser.add_argument(
    "--watermark_type",
    type=str,
    default="mb",
    choices=["mb", "noise", "gaussmark", "distilled"],
    help="Type of watermark to be used",
)


parser.add_argument(
    "--output_dir",
    type=str,
    default="lora",
    help="Directory to save the checkpoints",
)


parser.add_argument("--targeted", action="store_true",
                    help="Use targeted fine-tuning", default=False)

args = parser.parse_args()

model_name = args.model_name

model_suffix = model_name.split("/")[-1]
dataset_name = "Skylion007/openwebtext"
dataset_suffix = dataset_name.split("/")[-1]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",  # Load on one device temporarily
)


output_dir = f"output/{model_suffix}/{args.output_dir}"

if args.watermark_type == "mb" or args.watermark_type == "noise":
    finetuning_type = "targeted" if args.targeted else "full"
else:
    finetuning_type = "full"

seed = 12997009
if args.watermark_type == "distilled":
    watermark = KGWDistilled(model=base_model, tokenizer=tokenizer, gamma=0.25, kgw_device="cpu",
                             seeding_scheme="simple_1")

    config = {
        "gamma": 0.25,
        "kgw_device": "cpu",
        "seeding_scheme": "simple_1"
    }
elif args.watermark_type == "mb":
    k = 235
    final_matrix_path = f"saved_models/{dataset_suffix}_{model_suffix}/selector_matrix_k235.pth"
    final_weight = torch.load(final_matrix_path)
    delta = 1.2
    gamma = 0.25
    watermark = MbMark.mb(
        delta=delta,
        gamma=gamma,
        seed=seed,
        final_weight=final_weight,
        model=base_model,
        tokenizer=tokenizer,
        unembedding_param_name="lm_head",
        mode=Mode.Generate
    )
    config = {
        "delta": delta,
        "gamma": gamma,
        "num_clusters": final_weight.shape[0],
        "seed": seed,
    }

elif args.watermark_type == "noise":
    distribution = "gaussian"  # or "uniform"
    delta = 1.0
    watermark = MbMark.noise_injection(
        delta=delta,
        seed=seed,
        model=base_model,
        tokenizer=tokenizer,
        unembedding_param_name="lm_head",
        distribution=distribution,
        mode=Mode.Generate
    )
    config = {
        "seed": seed,
        "delta": delta,
        "distribution": distribution,
    }
elif args.watermark_type == "gaussmark":
    param = "model.layers.27.mlp.up_proj.weight"
    sigma = 0.045
    watermark = GaussMark(sigma=sigma, seed=seed,
                          target_param_name=param, tokenizer=tokenizer, model=base_model)

    config = {
        "sigma": sigma,
        "seed": seed,
        "target_param_name": param,
    }


watermarked_model = watermark.model


os.makedirs(output_dir, exist_ok=True)
max_steps = 3000
warmup_steps = 500
learning_rate = 1e-5
batch_size = 32
logging_steps = 50
save_steps = 500


# Load and shuffle dataset
raw_dataset = load_dataset(
    dataset_name, split="train[:2%]", trust_remote_code=True).shuffle(seed=42)


def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")


tokenized_dataset = raw_dataset.map(
    tokenize, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset, batch_size=batch_size,
                        collate_fn=data_collator, shuffle=True, num_workers=1)


if args.targeted:
    target_modules = ["lm_head"]
else:
    num_layers = len(watermarked_model.model.layers)
    target_modules = [f"model.layers.{i}.mlp.up_proj" for i in range(
        num_layers-10, num_layers)]
    target_modules += [f"model.layers.{i}.mlp.down_proj" for i in range(
        num_layers-10, num_layers)]
    target_modules += [f"model.layers.{i}.mlp.gate_proj" for i in range(
        num_layers-10, num_layers)]
    target_modules.append("lm_head")

# Apply LoRA to the unembedding layer
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(watermarked_model, lora_config)

# Optimizer and LR scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=max_steps,
)

# Training loop
model.train()
step = 0
progress_bar = tqdm(total=max_steps)


for batch in dataloader:
    batch = {k: v.to("cuda") for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    if step % logging_steps == 0:
        print(f"Step {step}: Loss {loss.item():.4f}")

    if step % save_steps == 0 and step > 0:
        model.save_pretrained(os.path.join(
            output_dir, f"checkpoint-step-{step}"))

    step += 1
    progress_bar.update(1)
    if step >= max_steps:
        break

progress_bar.close()

# Save the final model
model.save_pretrained(os.path.join(
    output_dir, f"checkpoint-step-{step}"))


if config:
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)
