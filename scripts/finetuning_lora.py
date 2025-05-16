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
from accelerate import FullyShardedDataParallelPlugin, Accelerator


torch.manual_seed(42)
# Configuration
accelerator = Accelerator()  # or "fp16" if needed


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


parser.add_argument("--targeted", action="store_true",
                    help="Use targeted fine-tuning", default=False)

args = parser.parse_args()

model_name = args.model_name

model_suffix = model_name.split("/")[-1]
dataset_name = "Skylion007/openwebtext"

if args.watermark_type == "mb":
    output_file = f"output/{model_suffix}/output_align=0_delta=1.2_gamma=0.3_k=4_seed=12997009_watermark=mb_dataset=realnewslike.json"
else:
    output_file = f"output/{model_suffix}/output_seed=12997009_watermark=mb2_dataset=realnewslike.json"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map={"": accelerator.process_index},  # Load on one device temporarily
)





with open(output_file, "r") as f:
    output_data = json.load(f)

output_dir = f"output/{model_suffix}/lora"
finetuning_type = "targeted" if args.targeted else "full"


watermark_type = output_data["watermark"]
config = output_data["config"]
if watermark_type == "mb":
    final_weight = torch.tensor(output_data["final_matrix"])
    watermark = MbMark.mb(
        delta=config["delta"],
        gamma=config["gamma"],
        seed=config["hash_key"],
        final_weight=final_weight,
        model=base_model,
        tokenizer=tokenizer,
        unembedding_param_name=config["unembedding_param_name"],
        mode=Mode.Generate,
    )
    checkpoint_suffix = f"ft={finetuning_type}_watermark=mb_delta={config['delta']}_gamma={config['gamma']}_k={config['n_clusters']}_seed={config['hash_key']}"

elif watermark_type == "mb2":
    watermark = MbMark.mb2(
        seed=config["hash_key"],
        model=base_model,
        tokenizer=tokenizer,
        unembedding_param_name=config["unembedding_param_name"],
        mode=Mode.Generate
    )
    checkpoint_suffix = f"watermark=mb2_seed={config['hash_key']}"


print(checkpoint_suffix)
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

model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, dataloader, lr_scheduler
)

# Training loop
model.train()
step = 0
progress_bar = tqdm(total=max_steps)


for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    if accelerator.is_main_process:
        if step % logging_steps == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")

    if step % save_steps == 0 and step > 0:
        if accelerator.is_main_process:
            model.save_pretrained(os.path.join(
                output_dir, f"checkpoint-{checkpoint_suffix}-step-{step}"))

    step += 1
    progress_bar.update(1)
    if step >= max_steps:
        break

progress_bar.close()

# Save the final model
if accelerator.is_main_process:
    model.save_pretrained(os.path.join(
        output_dir, f"checkpoint-{checkpoint_suffix}-step-{step}"))
