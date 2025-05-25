import argparse
from src.mbmark import MbMark, Mode
from src.gaussmark import GaussMark
from src.kgw_distilled import KGWDistilled
import os
import json
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()

    # Fixed defaults
    parser.add_argument('--prompt_length', type=int, default=50)
    parser.add_argument('--max_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--multinomial', action='store_true', default=True)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--watermark', type=str,
                        default="mb", choices=["mb", "gaussmark", "mb2", "mb3"])
    parser.add_argument('--dataset_path', type=str,
                        default="allenai/c4")
    parser.add_argument('--dataset_config_name', type=str,
                        default=None)
    parser.add_argument('--dataset_split', type=str,
                        default="validation")
    parser.add_argument('--data_field', type=str,
                        default="text")
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument('--step', type=int, default=500)
    parser.add_argument('--checkpoint_dir', type=str, required=True)

    args = parser.parse_args()

    return args


args = parse_args()
print(args)

# Check if the output file already exists
if os.path.exists(args.output_file):
    with open(args.output_file, "r") as f:
        output_data = json.load(f)
else:
    output_data = {}

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    device_map="auto")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                  device_map="auto", torch_dtype=torch.bfloat16)
base_model.eval()

device = base_model.device


prompt_dataset = load_dataset(
    args.dataset_path, args.dataset_config_name, split=args.dataset_split, trust_remote_code=True, streaming=args.streaming)

if args.dataset_path == "kmfoda/booksum":
    # Remove all columns except for the data_field
    prompt_dataset = prompt_dataset.remove_columns(
        [col for col in prompt_dataset.column_names if col != args.data_field])

# Shuffle the dataset with a fixed seed
prompt_dataset = prompt_dataset.shuffle(seed=42)
min_length = args.prompt_length + args.max_tokens


def filter_length(example):
    return len(tokenizer(example[args.data_field], truncation=True, max_length=min_length)["input_ids"]) >= min_length


def encode(examples):
    trunc_tokens = tokenizer(
        examples[args.data_field],
        truncation=True,
        padding=True,
        max_length=min_length,
        return_tensors="pt"
    ).to(device)
    examples["text"] = tokenizer.batch_decode(
        trunc_tokens["input_ids"], skip_special_tokens=True)
    prompt = tokenizer(
        examples["text"], truncation=True, padding=True, max_length=args.prompt_length, return_tensors="pt",
    ).to(device)
    examples["prompt_text"] = tokenizer.batch_decode(
        prompt["input_ids"], skip_special_tokens=True)
    examples["input_ids"] = prompt["input_ids"]
    examples["attention_mask"] = prompt["attention_mask"]
    examples["text_completion"] = tokenizer.batch_decode(
        trunc_tokens["input_ids"][:,
                                  args.prompt_length:], skip_special_tokens=True
    )
    return examples


prompt_dataset = prompt_dataset.filter(filter_length)
prompt_dataset = prompt_dataset.map(encode, batched=True)

dataloader = torch.utils.data.DataLoader(prompt_dataset, batch_size=32)

prompts = []
human_text = []
prompt_text = []
full_human_text = []
for batch in dataloader:
    if len(human_text) >= args.num_samples:
        break
    if (type(batch["input_ids"]) == list):
        batch["input_ids"] = torch.stack(batch["input_ids"], dim=1).to(device)
    if (type(batch["attention_mask"]) == list):
        batch["attention_mask"] = torch.stack(
            batch["attention_mask"], dim=1).to(device)
    prompts.append(batch)
    human_text.extend(batch["text_completion"])
    prompt_text.extend(batch["prompt_text"])
    full_human_text.extend(batch["text"])

human_text = human_text[:args.num_samples]
prompt_text = prompt_text[:args.num_samples]
full_human_text = full_human_text[:args.num_samples]

model_suffix = args.model_name.split("/")[-1]

# Open config.json file in checkpoint_dir
config_file = os.path.join(
    args.checkpoint_dir, "config.json")
with open(config_file, "r") as f:
    config_data = json.load(f)

seed = config_data["seed"]

if args.watermark == "mb":
    final_weight_file = config_data["final_weight_file"]
    with open(final_weight_file, "r") as f:
        json_data = json.load(f)
        final_weight = torch.tensor(json_data["final_matrix"])

    watermark = MbMark.mb(
        delta=config_data["delta"],
        gamma=config_data["gamma"],
        seed=seed,
        final_weight=final_weight,
        model=base_model,
        tokenizer=tokenizer,
        unembedding_param_name="lm_head",
        mode=Mode.Generate
    )

elif args.watermark == "mb2":
    watermark = MbMark.mb2(
        delta=config_data["delta"],
        seed=seed,
        model=base_model,
        tokenizer=tokenizer,
        unembedding_param_name="lm_head",
        mode=Mode.Generate
    )

elif args.watermark == "mb3":
    watermark = MbMark.mb3(
        delta=config_data["delta"],
        seed=seed,
        model=base_model,
        tokenizer=tokenizer,
        unembedding_param_name="lm_head",
        mode=Mode.Generate
    )
elif args.watermark == "gaussmark":
    param = config_data["target_param_name"]
    sigma = config_data["sigma"]
    watermark = GaussMark(sigma=sigma, seed=seed,
                          target_param_name=param, tokenizer=tokenizer, model=base_model)
elif args.watermark == "distilled":
    watermark = KGWDistilled(model=base_model, tokenizer=tokenizer)

watermarked_model = watermark.model

if args.step > 0:
    lora_ckpt_path = os.path.join(
        args.checkpoint_dir, f"checkpoint-step-{args.step}")

    peft_model = PeftModel.from_pretrained(watermarked_model, lora_ckpt_path)
    peft_model.merge_and_unload()
else:
    peft_model = watermarked_model

peft_model.eval()
model_text = []
full_model_text = []
for batch in tqdm(prompts):
    if len(model_text) >= args.num_samples:
        break
    with torch.no_grad():
        outputs = peft_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_tokens,
            min_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.multinomial,
            pad_token_id=tokenizer.eos_token_id
        )
        n_input_tokens = batch["input_ids"].shape[1]
        model_text.extend(tokenizer.batch_decode(
            outputs[:, n_input_tokens:], skip_special_tokens=True))
        full_model_text.extend(tokenizer.batch_decode(
            outputs, skip_special_tokens=True))

model_text = model_text[:args.num_samples]
full_model_text = full_model_text[:args.num_samples]

with torch.no_grad():
    del base_model, peft_model
    torch.cuda.empty_cache()

# Create dict
data = {
    "human_text": human_text,
    "prompt_text": prompt_text,
    "full_human_text": full_human_text,
    "model_text": model_text,
    "full_model_text": full_model_text
}

sample_data = {
    "samples": data,
    "model_name": args.model_name,
    "num_samples": args.num_samples,
    "temperature": args.temperature,
    "watermark": args.watermark,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "multinomial": args.multinomial,
    "prompt_length": args.prompt_length,
    "max_tokens": args.max_tokens,
    "vocab_size": len(tokenizer),
    "dataset_name": "{}-{}".format(args.dataset_path, args.dataset_config_name),
}

if args.watermark == "mb":
    sample_data["final_matrix"] = json_data["final_matrix"]


if args.watermark == "mb":
    config = {
        "gamma": config_data["gamma"],
        "delta": config_data["delta"],
        "hash_key": seed,
        "n_clusters": final_weight.size(0),
        "unembedding_param_name": "lm_head",
    }
elif args.watermark in ["mb2", "mb3"]:
    config = {
        "hash_key": seed,
        "delta": config_data["delta"],
        "unembedding_param_name": "lm_head",
    }
elif args.watermark == "gaussmark":
    config = {
        "sigma": config_data["sigma"],
        "hash_key": seed,
        "target_param_name": config_data["target_param_name"],
    }


sample_data["config"] = config


output_data.update(sample_data)


with open(args.output_file, "w") as f:
    json.dump(output_data, f, indent=4)
