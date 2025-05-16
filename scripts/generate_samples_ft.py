import argparse
from src.mbmark import MbMark, Mode
from src.gaussmark import GaussMark
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
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--hash_key', type=int, default=15485863,
                        help="PRF for the watermarking matrix")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--watermark', type=str,
                        default="mb", choices=["mb", "gaussmark", "mb2"])
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


prompt_dataset = load_dataset(
    args.dataset_path, args.dataset_config_name, split=args.dataset_split, trust_remote_code=True, streaming=args.streaming)

if args.dataset_path == "kmfoda/booksum":
    # Remove all columns except for the data_field
    prompt_dataset = prompt_dataset.remove_columns(
        [col for col in prompt_dataset.column_names if col != args.data_field])

# Shuffle the dataset with a fixed seed
prompt_dataset = prompt_dataset.shuffle(seed=args.seed)
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

if args.watermark == "mb":
    lora_ckpt_path = f"output/{model_suffix}/lora/checkpoint-watermark=mb_delta=1.2_gamma=0.3_k=4_seed=12997009-step-{args.step}"
elif args.watermark == "mb2":
    lora_ckpt_path = f"output/{model_suffix}/lora/checkpoint-watermark=mb2_seed=12997009-step-{args.step}"

original_unembedding = model.lm_head.weight.data.clone()
peft_model = PeftModel.from_pretrained(model, lora_ckpt_path)
peft_model.merge_and_unload()
updated_unembedding = peft_model.lm_head.weight.data.clone()
with torch.no_grad():
    model.lm_head.weight.data.copy_(original_unembedding)


mb_mark = MbMark.from_augmented(
        updated_unembedding, model, tokenizer, unembedding_param_name="lm_head", mode=Mode.Generate)

watermarked_model = mb_mark.model


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


if args.watermark_type == "mb":
    orig_output_file = f"output/{model_suffix}/output_align=0_delta=1.2_gamma=0.3_k=4_seed=12997009_watermark=mb_dataset=realnewslike.json"
else:
    orig_output_file = f"output/{model_suffix}/output_seed=12997009_watermark=mb2_dataset=realnewslike.json"


with open(orig_output_file, "r") as f:
    orig_data = json.load(f)
    config = orig_data["config"]



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
    "dataset_name": "{}-{}".format(args.dataset_path, args.dataset_config_name),
}

if args.watermark == "mb":
    sample_data["final_matrix"] = orig_data["final_matrix"]


output_data.update(sample_data)


with open(args.output_file, "w") as f:
    json.dump(output_data, f, indent=4)
