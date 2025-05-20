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
    choices=["mb", "mb2"],
    help="Type of watermark to be used",
)

# Add argument for number of clustest which is a required parameter if watermark_type = mb
parser.add_argument(
    "--num_clusters",
    type=int,
    default=4,
    help="Number of clusters to be used for mb watermarking",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="lora",
    help="Directory to save the output model",
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
finetuning_type = "targeted" if args.targeted else "full"


seed = 12997009

if args.watermark_type == "mb":
    final_weight_file = f"saved_models/{dataset_suffix}_{model_suffix}/final_weights_k{args.num_clusters}.json"
    with open(final_weight_file, "r") as f:
        json_data = json.load(f)
        final_weight = torch.tensor(json_data["final_matrix"])

    watermark = MbMark.mb(
        delta=1.2,
        gamma=0.3,
        seed=seed,
        final_weight=final_weight,
        model=base_model,
        tokenizer=tokenizer,
        unembedding_param_name="lm_head",
        mode=Mode.Generate,
        low_rank=False
    )

elif args.watermark_type == "mb2":
    watermark = MbMark.mb2(
        seed=seed,
        model=base_model,
        tokenizer=tokenizer,
        unembedding_param_name="lm_head",
        mode=Mode.Generate
    )


watermarked_model = watermark.model

os.makedirs(output_dir, exist_ok=True)
max_steps = 2500
warmup_steps = 500
learning_rate = 1e-5
batch_size = 32
logging_steps = 50
save_steps = 500


# Load and shuffle dataset
raw_dataset = load_dataset(dataset_name, split="train[:1%]").shuffle(seed=42)


def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")


tokenized_dataset = raw_dataset.map(
    tokenize, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset, batch_size=batch_size,
                        collate_fn=data_collator, shuffle=True, num_workers=2)


if args.targeted:
    target_modules = ["lm_head"]
else:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", "lm_head"]

# Apply LoRA to the unembedding layer
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


if torch.cuda.device_count() > 1:
    watermarked_model.is_parallelizable = True
    watermarked_model.model_parallel = True

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
    batch = batch.to("cuda")
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
