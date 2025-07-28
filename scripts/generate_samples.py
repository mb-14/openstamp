import argparse
from src.mbmark import MbMark, Mode
from src.gaussmark import GaussMark
from src.kgwmark import KGWMark
from src.kgw_distilled import KGWDistilled
import os
import json
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, LogitsProcessorList
from torch.utils.data import TensorDataset


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
                        default="mb", choices=["mb", "gaussmark", "mb2", "mb3", "noise", "distilled", "kgw", "kgw_llr"])
    parser.add_argument('--distribution', type=str, default="symmetric_beta",
                        choices=["symmetric_beta", "gaussian",
                                 "uniform", "hidden_states", "truncated_normal", "low_rank"],
                        help="Distribution to sample the offset matrix from")
    parser.add_argument('--dataset_path', type=str,
                        default="allenai/c4")
    parser.add_argument('--dataset_config_name', type=str,
                        default=None)
    parser.add_argument('--dataset_split', type=str,
                        default="validation")
    parser.add_argument('--data_field', type=str,
                        default="text")
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--sigma", type=float, default=0.008,
                        help="Standard deviation for GaussMark")
    parser.add_argument("--target_param_name", type=str,
                        default="model.layers.27.mlp.up_proj.weight",)
    parser.add_argument("--k", type=int, default=16,
                        help="Number of clusters for the selector matrix in MbMark")

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
watermarked_model = None
watermarked_processor = None
if args.watermark == "mb":
    # Load final weights into a torch tensor
    dataset_suffix = "openwebtext"
    model_suffix = args.model_name.split("/")[-1]
    final_matrix_path = f"saved_models/{dataset_suffix}_{model_suffix}/selector_matrix_k{args.k}.pth"
    final_weight = torch.load(final_matrix_path)
    mb_mark = MbMark.mb(
        delta=args.delta,
        gamma=args.gamma,
        seed=args.hash_key,
        final_weight=final_weight,
        model=model,
        unembedding_param_name="lm_head",
        tokenizer=tokenizer,
        mode=Mode.Generate,
    )
    watermarked_model = mb_mark.model
elif args.watermark == "noise":
    mb_mark = MbMark.noise_injection(
        delta=args.delta,
        seed=args.hash_key,
        model=model,
        unembedding_param_name="lm_head",
        tokenizer=tokenizer,
        distribution=args.distribution,
        mode=Mode.Generate
    )

    watermarked_model = mb_mark.model
elif args.watermark == "gaussmark":
    # target_param_name = "model.layers.20.mlp.up_proj.weight"
    target_param_name = args.target_param_name
    sigma = args.sigma
    gaussmark = GaussMark(sigma=sigma, seed=args.hash_key,
                          target_param_name=target_param_name, tokenizer=tokenizer, model=model)
    watermarked_model = gaussmark.model
elif args.watermark == "distilled":
    watermark = KGWDistilled(model=model, tokenizer=tokenizer, gamma=0.25,
                             seeding_scheme="simple_1", kgw_device="cpu")
    watermarked_model = watermark.model
elif args.watermark == "kgw" or args.watermark == "kgw_llr":
    kgw_device = device
    watermark = KGWMark(model=model, tokenizer=tokenizer, gamma=args.gamma,
                        delta=args.delta, hash_key=args.hash_key, kgw_device=kgw_device)
    watermarked_processor = watermark.watermark

model_text = []
full_model_text = []
for batch in tqdm(prompts):
    if len(model_text) >= args.num_samples:
        break
    with torch.no_grad():
        if watermarked_model is not None:
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
        elif watermarked_processor is not None:
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_tokens,
                min_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=args.multinomial,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([watermarked_processor])
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
        "target_param_name": target_param_name
    }
elif args.watermark == "noise":
    config = {
        "hash_key": args.hash_key,
        "distribution": args.distribution,
        "delta": args.delta,
        "unembedding_param_name": "lm_head",
    }
elif args.watermark == "distilled":
    config = {
        "gamma": 0.25,
        "seeding_scheme": "simple_1",
        "kgw_device": "cpu",
    }
elif args.watermark == "kgw" or args.watermark == "kgw_llr":
    config = {
        "hash_key": args.hash_key,
        "kgw_device": str(kgw_device),
        "gamma": args.gamma,
        "delta": args.delta,
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
    "dataset_name": "{}-{}".format(args.dataset_path, args.dataset_config_name),
}


output_data.update(sample_data)


with open(args.output_file, "w") as f:
    json.dump(output_data, f, indent=4)
