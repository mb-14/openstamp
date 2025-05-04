import argparse
from src.utils import MbMark, GaussMark
import os
import json
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from torch.utils.data import TensorDataset, DataLoader


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
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--hash_key', type=int, default=15485863,
                        help="PRF for the watermarking matrix")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--watermark', type=str,
                        default="mb", choices=["mb", "gaussmark"])

    args = parser.parse_args()

    return args


args = parse_args()
print(args)

with open(args.output_file, "r") as f:
    output_data = json.load(f)

set_seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    device_map="auto")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                             device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

device = model.device


realnewslike = load_dataset("allenai/c4", "realnewslike", split="validation")
# Shuffle the dataset with a fixed seed
realnewslike = realnewslike.shuffle(seed=args.seed)
min_length = args.prompt_length + args.max_tokens


def filter_length(example):
    return len(tokenizer(example["text"], truncation=True, max_length=min_length)["input_ids"]) >= min_length


def encode(examples):
    trunc_tokens = tokenizer(
        examples["text"],
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


realnewslike = realnewslike.filter(filter_length)
realnewslike = realnewslike.map(encode, batched=True)

dataloader = torch.utils.data.DataLoader(realnewslike, batch_size=32)

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

if args.watermark == "mb":
    # Load final weights into a torch tensor
    final_weight = torch.tensor(output_data["final_matrix"])
    mb_mark = MbMark(
        delta=args.delta,
        gamma=args.gamma,
        seed=args.hash_key,
        final_weight=final_weight,
        model=model,
        unembedding_param_name="lm_head",
        tokenizer=tokenizer,
        mode="generate"
        )
    watermarked_model = mb_mark.model
elif args.watermark == "gaussmark":
    param = "model.layers.27.mlp.up_proj.weight"
    sigma = 0.04
    gaussmark = GaussMark(sigma=sigma, seed=args.hash_key,
                          target_param_name=param, tokenizer=tokenizer, model=model)
    watermarked_model = gaussmark.model

model_text = []
full_model_text = []
for batch in tqdm(prompts):
    if len(model_text) >= args.num_samples:
        break
    with torch.no_grad():
        outputs = watermarked_model.generate(
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
    del model
    torch.cuda.empty_cache()

# Create dict
data = {
    "human_text": human_text,
    "prompt_text": prompt_text,
    "full_human_text": full_human_text,
    "model_text": model_text,
    "full_model_text": full_model_text
}
if args.watermark == "mb":
    config = {
        "gamma": args.gamma,
        "delta": args.delta,
        "hash_key": args.hash_key,
        "n_clusters": final_weight.size(0),
        "unembedding_param_name": "lm_head",
    }
elif args.watermark == "gaussmark":
    config = {
        "sigma": sigma,
        "hash_key": args.hash_key,
        "target_param_name": "model.layers.27.mlp.up_proj.weight",
    }

sample_data = {
    "samples": data,
    "model_name": args.model_name,
    "num_samples": args.num_samples,
    "temperature": args.temperature,
    "watermark": args.watermark,
    "config": config,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "multinomial": args.multinomial,
    "prompt_length": args.prompt_length,
    "max_tokens": args.max_tokens,
    "vocab_size": len(tokenizer),
    "dataset_name": "realnewslike",
}



output_data.update(sample_data)


with open(args.output_file, "w") as f:
    json.dump(output_data, f, indent=4)