import argparse
from src.mbmark import MbMark, Mode
from src.gaussmark import GaussMark
from src.kgwmark import KGWMark
from src.kgw_distilled import KGWDistilled
import os
import json
from pathlib import Path
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, LogitsProcessorList, AutoConfig
from torch.utils.data import TensorDataset
from src.rl_watermark.ds_utils import convert_linear_layer_to_lora
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
    parser.add_argument('--hash_key', type=int, default=15485863,
                        help="PRF for the watermarking matrix")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--watermark', type=str,
                        default="mb", choices=["mb", "mb_binom", "gaussmark", "noise", "distilled", "kgw", "kgw_llr", "rl", "binoc", "mb_discrete"],)
    parser.add_argument('--distribution', type=str, default="symmetric_beta",
                        choices=["symmetric_beta", "gaussian",
                                 "uniform", "hidden_states", "truncated_normal", "low_rank"],
                        help="Distribution to sample the offset matrix from")
    parser.add_argument('--dataset', type=str,
                        default="realnewslike", choices=["realnewslike", "wikipedia", "arxiv", "booksum", "combined"])

    parser.add_argument("--sigma", type=float, default=0.008,
                        help="Standard deviation for GaussMark")
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
    parser.add_argument('--binoc_performer_lora_path', type=str,
                        help="Local path to the Binocular performer LoRA weights", default=None)
    parser.add_argument('--binoc_observer_lora_path', type=str,
                        help="Local path to the Binocular observer LoRA weights", default=None)
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help="Directory containing the LoRA checkpoints")
    parser.add_argument('--step', type=int, default=0,
                        help="Step of the LoRA checkpoint to load. If 0, no LoRA is applied.")

    args = parser.parse_args()

    return args


dataset_registry = {
    "realnewslike": {
        "path": "allenai/c4",
        "config": "realnewslike",
        "split": "validation",
        "data_field": "text",
        "streaming": False,
    },
    "wikipedia": {
        "path": "wikipedia",
        "config": "20220301.en",
        "split": "train",
        "data_field": "text",
        "streaming": True,
    },
    "arxiv": {
        "path": "armanc/scientific_papers",
        "config": "arxiv",
        "split": "test",
        "data_field": "article",
        "streaming": False,
    },
    "booksum": {
        "path": "kmfoda/booksum",
        "config": None,
        "split": "test",
        "data_field": "chapter",
        "streaming": False,
    },
}

args = parse_args()
print(args)

# Check if the output file already exists
if os.path.exists(args.output_file):
    with open(args.output_file, "r") as f:
        output_data = json.load(f)
else:
    output_data = {}

set_seed(args.seed)


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
    model, tokenizer = load_model(args.model_name)


device = model.device
if args.dataset == "combined":
    selected_keys = ["realnewslike", "wikipedia", "arxiv", "booksum"]
else:
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
    dataset = load_dataset(
        spec["path"],
        spec["config"],
        split=spec["split"],
        streaming=spec["streaming"],
        trust_remote_code=True,
    )

    # Reduce to necessary field (booksum special case)
    if key == "booksum":
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col != spec["data_field"]])

    # Shuffle with buffer
    dataset = dataset.shuffle(seed=args.seed)

    dataset = dataset.filter(lambda x: filter_length(x, spec["data_field"]))

    # Collect samples
    sample_buffer = []
    for example in dataset:
        encoded = encode(example, spec["data_field"])
        sample_buffer.append(encoded)
        if len(sample_buffer) >= samples_per_dataset:
            break

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
watermarked_model = None
watermarked_processor = None
temperature = args.temperature

if args.watermark in ["mb", "mb_binom", "mb_discrete"]:
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
elif args.watermark == "rl":
    watermarked_model = convert_linear_layer_to_lora(
        model, part_module_name='decoder.layers.', lora_dim=128)
    watermarked_model.load_state_dict(torch.load(
        args.rl_model_path+"/pytorch_model.bin", map_location='cpu'))
    watermarked_model = watermarked_model.cuda()
    watermarked_model.eval()
elif args.watermark == "binoc":
    if args.binoc_performer_lora_path is None:
        raise ValueError(
            "Please provide the path to the Binocular performer LoRA weights using --binoc_lora_path")
    watermarked_model = PeftModel.from_pretrained(
        model, args.binoc_performer_lora_path)
    watermarked_model.eval()

if args.step > 0:
    lora_ckpt_path = os.path.join(
        args.checkpoint_dir, f"checkpoint-step-{args.step}")

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
    "full_model_text": full_model_text
}
if args.watermark in ["mb", "mb_binom", "mb_discrete"]:
    semalign = bool(selector_metrics.get("sem_align", False)
                    ) if selector_metrics else False
    embedding_model = selector_metrics.get(
        "embedding_model") if selector_metrics else None
    align_method = selector_metrics.get(
        "align_method") if selector_metrics else None
    config = {
        "gamma": args.gamma,
        "delta": args.delta,
        "hash_key": args.hash_key,
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
elif args.watermark == "rl":
    config = {
        "rl_model_path": args.rl_model_path,
    }
elif args.watermark == "binoc":
    config = {
        "observer_lora_path": args.binoc_observer_lora_path,
        "performer_lora_path": args.binoc_performer_lora_path,
    }
sample_data = {
    "samples": data,
    "model_name": args.model_name,
    "num_samples": args.num_samples,
    "temperature": temperature,
    "watermark": args.watermark,
    "config": config,
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
