import argparse
from src.openstamp import OpenStamp, Mode
from src.christmark import ChristMark
from src.gaussmark import GaussMark
from src.kgwmark import KGWMark
from src.kgw_distilled import KGWDistilled
from src.unigramwm import Unigram
import os
import json
from pathlib import Path
from datasets import Dataset
import aiohttp
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, LogitsProcessorList, AutoConfig
from torch.utils.data import TensorDataset
from src.rl_watermark.ds_utils import convert_linear_layer_to_lora
from src.dataset_registry import dataset_registry, load_registry_dataset
from src.alpaca_split import (
    encode_instruction,
    filter_alpaca_eval_dataset,
    filter_instruction_prompts,
)
from src.utils import load_model
import random
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
    parser.add_argument('--watermark_seed', type=int, default=15485863,
                        help="PRF for the watermarking matrix")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--generation_seed', type=int, default=42)
    parser.add_argument('--watermark', type=str,
                        default="openstamp", choices=["openstamp", "openstamp_binom", "openstamp_discrete", "gaussmark", "unremovable", "christ", "noise", "distilled", "kgw", "kgw_llr", "rl", "unigram"],)
    parser.add_argument('--distribution', type=str, default="symmetric_beta",
                        choices=["symmetric_beta", "gaussian",
                                 "uniform", "hidden_states", "truncated_normal", "low_rank"],
                        help="Distribution to sample the offset matrix from")
    parser.add_argument(
        '--dataset',
        type=str,
        default="realnewslike",
        choices=sorted(dataset_registry.keys()),
    )

    parser.add_argument("--sigma", type=float, default=0.008,
                        help="Standard deviation for GaussMark")
    parser.add_argument("--epsilon", type=float, default=0.5,
                        help="Standard deviation of Unremovable (Christ et al.) lm_head.bias key")
    parser.add_argument("--target_param_name", type=str,
                        default="model.layers.27.mlp.up_proj.weight",)
    parser.add_argument(
        "--selector_matrix_dir",
        type=str,
        default=None,
        help="Directory containing selector_matrix.pth and selector_metrics.json",
    )
    parser.add_argument("--rl_model_path", type=str,
                        help="Local path to the RL model", default=None)
    parser.add_argument('--checkpoint_dir', type=str, required=False,
                        help="Directory containing the LoRA checkpoints")
    parser.add_argument('--step', type=int, default=0,
                        help="Step of the LoRA checkpoint to load. If 0, no LoRA is applied.")
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "nf4", "int8"],
        help="bitsandbytes load-time quantization for generation (none/nf4/int8)",
    )

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

set_seed(args.generation_seed)


if args.watermark == "rl":
    model_config = AutoConfig.from_pretrained(args.model_name)
    for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
        if hasattr(model_config, key):
            setattr(model_config, key, 0.0)
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 config=model_config, device_map="auto", torch_dtype=torch.bfloat16).train()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
else:
    model, tokenizer = load_model(
        args.model_name,
        quantization=None if args.quantization == "none" else args.quantization,
    )


device = model.device
selected_keys = [args.dataset]

samples_per_dataset = args.num_samples // len(selected_keys)
min_length = args.prompt_length + args.max_tokens


def filter_length(example, field):
    return len(tokenizer(example[field], truncation=True, max_length=min_length)["input_ids"]) >= min_length


def encode(example, field):
    trunc_tokens = tokenizer(
        example[field],
        truncation=True,
        padding=True,
        max_length=min_length,
        return_tensors="pt"
    ).to(device)
    text = tokenizer.batch_decode(
        trunc_tokens["input_ids"], skip_special_tokens=True)[0]

    prompt = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=args.prompt_length,
        return_tensors="pt"
    ).to(device)

    prompt_text = tokenizer.batch_decode(
        prompt["input_ids"], skip_special_tokens=True)[0]

    if args.model_name == "microsoft/phi-4":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "Tokenizer does not support chat templates, but microsoft/phi-4 requires it.")
        instruction = (
            "Continue the text exactly from where it ends. "
            "Do not add preambles, summaries, or extra commentary. "
            "Output only the direct continuation.\n\n"
            f"{prompt_text}\n\nContinuation:"
        )
        messages = [{"role": "user", "content": instruction}]
        chat_prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding="max_length",
            max_length=args.prompt_length + 100,
            return_tensors="pt"
        ).to(device)
        input_ids = prompt_inputs.squeeze(0)
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = prompt["input_ids"].squeeze(0)
        attention_mask = prompt["attention_mask"].squeeze(0)
        chat_prompt_text = prompt_text

    return {
        "text": text,
        "prompt_text": prompt_text,
        "chat_prompt_text": chat_prompt_text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text_completion": tokenizer.batch_decode(
            trunc_tokens["input_ids"][:, args.prompt_length:], skip_special_tokens=True)[0],
    }


all_samples = []

for key in selected_keys:
    spec = dataset_registry[key]

    # Load dataset
    dataset = load_registry_dataset(
        spec,
        trust_remote_code=True,
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
    )

    prompt_mode = spec.get("prompt_mode", "continuation")
    data_field = spec["data_field"]
    completion_field = spec.get("completion_field")

    # Reduce to necessary field (booksum special case)
    if key == "booksum":
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col != data_field])

    if prompt_mode == "instruction":
        if not completion_field:
            raise ValueError(
                f"dataset {key} uses prompt_mode=instruction but has no completion_field"
            )
        # AlpacaEval: filter gold≥50 tokens first, then shuffle seed 42, take first N.
        if key == "alpaca_eval":
            dataset = filter_alpaca_eval_dataset(
                dataset, tokenizer, completion_field=completion_field
            )
        dataset = filter_instruction_prompts(
            dataset,
            tokenizer,
            prompt_field=data_field,
            completion_field=completion_field,
            prompt_length=args.prompt_length,
        )
        dataset = dataset.shuffle(seed=args.generation_seed)
    else:
        # Shuffle with buffer
        dataset = dataset.shuffle(seed=args.generation_seed)
        dataset = dataset.filter(
            lambda x, field=data_field: filter_length(x, field)
        )

    # Collect samples
    sample_buffer = []
    for example in dataset:
        if prompt_mode == "instruction":
            encoded = encode_instruction(
                example,
                tokenizer,
                prompt_field=data_field,
                completion_field=completion_field,
                prompt_length=args.prompt_length,
                device=device,
            )
        else:
            encoded = encode(example, data_field)
        sample_buffer.append(encoded)
        if len(sample_buffer) >= samples_per_dataset:
            break

    if len(sample_buffer) < samples_per_dataset:
        raise RuntimeError(
            f"Only collected {len(sample_buffer)}/{samples_per_dataset} samples "
            f"from dataset={key} (prompt_mode={prompt_mode}). "
            "Relax filters or reduce num_samples."
        )

    all_samples.extend(sample_buffer)

# --- Final shuffle with local RNG ---

combined_dataset = Dataset.from_list(all_samples)

dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=32)


prompts = []
human_text = []
prompt_text = []
chat_prompt_text = []
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
    chat_prompt_text.extend(batch["chat_prompt_text"])
    full_human_text.extend(batch["text"])

human_text = human_text[:args.num_samples]
prompt_text = prompt_text[:args.num_samples]
chat_prompt_text = chat_prompt_text[:args.num_samples]
full_human_text = full_human_text[:args.num_samples]
# Original dataset completions (may be short for instruction data).
reference_text = list(human_text)
instruction_mode = any(
    dataset_registry[k].get("prompt_mode") == "instruction" for k in selected_keys
)
watermarked_model = None
watermarked_processor = None
temperature = args.temperature

# Instruction prompts have short reference answers that break LLR null scoring
# (division by near-zero length). Use same-length unwatermarked generations instead.
if instruction_mode:
    print(
        "Instruction dataset: generating unwatermarked null completions "
        f"(len={args.max_tokens}) for human_text"
    )
    null_model = model
    if args.step > 0:
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir is required when --step > 0")
        lora_ckpt_path = os.path.join(
            args.checkpoint_dir, f"checkpoint-{args.step}")
        if "_config2" in args.checkpoint_dir:
            null_model = AutoModelForCausalLM.from_pretrained(
                lora_ckpt_path, device_map="auto", torch_dtype=torch.bfloat16
            ).eval()
        else:
            peft_model = PeftModel.from_pretrained(model, lora_ckpt_path)
            peft_model.merge_and_unload()
            null_model = peft_model.eval().to(device)

    human_text = []
    for batch in tqdm(prompts):
        if len(human_text) >= args.num_samples:
            break
        with torch.no_grad():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            if temperature == 0.0:
                do_sample = False
            else:
                do_sample = args.multinomial
            outputs = null_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_tokens,
                min_new_tokens=args.max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
            n_input_tokens = batch["input_ids"].shape[1]
            batch_continuations = tokenizer.batch_decode(
                outputs[:, n_input_tokens:], skip_special_tokens=True)
            if args.model_name == "microsoft/phi-4":
                batch_continuations = [
                    (t[len("assistant"):].lstrip() if t.startswith("assistant") else t)
                    for t in batch_continuations
                ]
            human_text.extend(batch_continuations)
    human_text = human_text[:args.num_samples]

    with torch.no_grad():
        if null_model is not model:
            del null_model
        torch.cuda.empty_cache()
    # Watermark embedding must start from a clean base model.
    model, tokenizer = load_model(
        args.model_name,
        quantization=None if args.quantization == "none" else args.quantization,
    )
    device = model.device

if args.watermark in ["openstamp", "openstamp_binom", "openstamp_discrete"]:
    # Load final weights into a torch tensor
    selector_metrics = None
    selector_metrics_path = None
    if not args.selector_matrix_dir:
        raise ValueError(
            "For MB watermarking, please provide --selector_matrix_dir."
        )
    selector_dir = Path(args.selector_matrix_dir)
    final_matrix_path = selector_dir / "selector_matrix.pth"
    selector_metrics_path = selector_dir / "selector_metrics.json"
    if selector_metrics_path.exists():
        with open(selector_metrics_path, "r") as f:
            selector_metrics = json.load(f)
    final_weight = torch.load(final_matrix_path)
    openstamp_mark = OpenStamp.from_config(
        delta=args.delta,
        gamma=args.gamma,
        seed=args.watermark_seed,
        final_weight=final_weight,
        model=model,
        unembedding_param_name="lm_head",
        tokenizer=tokenizer,
        mode=Mode.Generate,
    )
    watermarked_model = openstamp_mark.model
elif args.watermark == "noise":
    openstamp_mark = OpenStamp.noise_injection(
        delta=args.delta,
        seed=args.watermark_seed,
        model=model,
        unembedding_param_name="lm_head",
        tokenizer=tokenizer,
        distribution=args.distribution,
        mode=Mode.Generate
    )

    watermarked_model = openstamp_mark.model
elif args.watermark == "gaussmark":
    target_param_name = args.target_param_name
    sigma = args.sigma
    gaussmark = GaussMark(sigma=sigma, seed=args.watermark_seed,
                          target_param_name=target_param_name, tokenizer=tokenizer, model=model)
    watermarked_model = gaussmark.model
elif args.watermark in {"unremovable", "christ"}:
    christ = ChristMark(
        epsilon=args.epsilon,
        seed=args.watermark_seed,
        tokenizer=tokenizer,
        model=model,
    )
    watermarked_model = christ.model
elif args.watermark == "distilled":
    watermark = KGWDistilled(model=model, tokenizer=tokenizer, gamma=args.gamma,
                             delta=args.delta, seeding_scheme="simple_1", hash_key=args.watermark_seed, kgw_device="cpu")
    watermarked_model = watermark.model
elif args.watermark == "kgw" or args.watermark == "kgw_llr":
    kgw_device = device
    watermark = KGWMark(model=model, tokenizer=tokenizer, gamma=args.gamma,
                        delta=args.delta, hash_key=args.watermark_seed, kgw_device=kgw_device)
    watermarked_processor = watermark.watermark
elif args.watermark == "unigram":
    watermark = Unigram(gamma=args.gamma, delta=args.delta, hash_key=args.watermark_seed, tokenizer=tokenizer)
    watermarked_processor = watermark.watermark
elif args.watermark == "rl":
    watermarked_model = convert_linear_layer_to_lora(
        model, part_module_name='decoder.layers.', lora_dim=128)
    watermarked_model.load_state_dict(torch.load(
        args.rl_model_path+"/pytorch_model.bin", map_location='cpu'))
    watermarked_model = watermarked_model.cuda()
    watermarked_model.eval()

if args.step > 0:
    if not args.checkpoint_dir:
        raise ValueError("--checkpoint_dir is required when --step > 0")
    lora_ckpt_path = os.path.join(
        args.checkpoint_dir, f"checkpoint-{args.step}")
    if "_config2" in args.checkpoint_dir:
        # Config 2 is full-finetuning, so we load the entire model
        watermarked_model = AutoModelForCausalLM.from_pretrained(lora_ckpt_path, device_map="auto", torch_dtype=torch.bfloat16).eval()
    else:
        peft_model = PeftModel.from_pretrained(watermarked_model, lora_ckpt_path)
        peft_model.merge_and_unload()
        watermarked_model = peft_model.eval().to(device)

model_text = []
full_model_text = []


def strip_phi4_assistant_prefix(text: str) -> str:
    if text.startswith("assistant"):
        return text[len("assistant"):].lstrip()
    return text


for batch in tqdm(prompts):
    if len(model_text) >= args.num_samples:
        break
    with torch.no_grad():
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if temperature == 0.0:
            do_sample = False
        else:
            do_sample = args.multinomial
        if watermarked_model is not None:
            outputs = watermarked_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_tokens,
                min_new_tokens=args.max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        elif watermarked_processor is not None:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_tokens,
                min_new_tokens=args.max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([watermarked_processor])
            )

        n_input_tokens = batch["input_ids"].shape[1]
        batch_continuations = tokenizer.batch_decode(
            outputs[:, n_input_tokens:], skip_special_tokens=True)
        if args.model_name == "microsoft/phi-4":
            batch_continuations = [strip_phi4_assistant_prefix(
                t) for t in batch_continuations]
        model_text.extend(batch_continuations)
        if args.model_name == "microsoft/phi-4":
            batch_size = outputs.shape[0]
            for i in range(batch_size):
                full_model_text.append(" " +
                                       f"{batch['prompt_text'][i]}{batch_continuations[i]}"
                                       )
        else:
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
    "chat_prompt_text": chat_prompt_text,
    "full_human_text": full_human_text,
    "model_text": model_text,
    "full_model_text": full_model_text,
}
if instruction_mode:
    data["reference_text"] = reference_text
if args.watermark in ["openstamp", "openstamp_binom", "openstamp_discrete"]:
    semalign = bool(selector_metrics.get("sem_align", False)
                    ) if selector_metrics else False
    embedding_model = selector_metrics.get(
        "embedding_model") if selector_metrics else None
    align_method = selector_metrics.get(
        "align_method") if selector_metrics else None
    config = {
        "gamma": args.gamma,
        "delta": args.delta,
        "watermark_seed": args.watermark_seed,
        "n_clusters": final_weight.size(0),
        "unembedding_param_name": "lm_head",
        "semalign": semalign,
        "embedding_model": embedding_model if semalign else None,
        "align_method": align_method if semalign else None,
        "selector_matrix_dir": str(selector_dir),
        "selector_metrics_path": str(selector_metrics_path) if selector_metrics_path and selector_metrics_path.exists() else None,
    }
elif args.watermark == "gaussmark":
    config = {
        "sigma": sigma,
        "watermark_seed": args.watermark_seed,
        "target_param_name": target_param_name
    }
elif args.watermark in {"unremovable", "christ"}:
    config = {
        "epsilon": args.epsilon,
        "watermark_seed": args.watermark_seed,
        "vocab_size": christ.vocab_size,
        "quantization": args.quantization,
    }
elif args.watermark == "noise":
    config = {
        "watermark_seed": args.watermark_seed,
        "distribution": args.distribution,
        "delta": args.delta,
        "unembedding_param_name": "lm_head",
    }
elif args.watermark == "distilled":
    config = {
        "gamma": args.gamma,
        "delta": args.delta,
        "seeding_scheme": "simple_1",
        "kgw_device": "cpu",
        "watermark_seed": args.watermark_seed,
    }
elif args.watermark == "kgw" or args.watermark == "kgw_llr":
    config = {
        "watermark_seed": args.watermark_seed,
        "kgw_device": str(kgw_device),
        "gamma": args.gamma,
        "delta": args.delta,
    }
elif args.watermark == "unigram":
    config = {
        "gamma": args.gamma,
        "delta": args.delta,
        "watermark_seed": args.watermark_seed,
    }
elif args.watermark == "rl":
    config = {
        "rl_model_path": args.rl_model_path,
    }
sample_data = {
    "samples": data,
    "model_name": args.model_name,
    "num_samples": args.num_samples,
    "temperature": temperature,
    "watermark": args.watermark,
    "config": config,
    "quantization": args.quantization,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "multinomial": args.multinomial,
    "prompt_length": args.prompt_length,
    "max_tokens": args.max_tokens,
    "vocab_size": len(tokenizer),
    "dataset_name": args.dataset,
}


output_data.update(sample_data)


with open(args.output_file, "w") as f:
    json.dump(output_data, f, indent=4)
