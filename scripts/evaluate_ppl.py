from tqdm import tqdm
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import numpy as np


def compute_ppl(samples_dict, batch_size, column_name, oracle_model, oracle_tokenizer, original_tokenizer):
    print("Computing PPL for ", column_name)
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    device = oracle_model.device

    samples = samples_dict[column_name]
    prompts = samples_dict["prompt_text"]

    for i in range(0, len(samples), batch_size):

        s = samples[i:i + batch_size]
        encodings = oracle_tokenizer(
            s,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_batch = encodings["input_ids"]
        attn_mask = encodings["attention_mask"]

        labels = encoded_batch

        with torch.no_grad():
            out_logits = oracle_model(
                encoded_batch, attention_mask=attn_mask).logits

        prompt_text = prompts[i:i + batch_size]
        prompt_encodings = original_tokenizer(
            prompt_text,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        prompt_attn_mask = prompt_encodings["attention_mask"]

        # match shape of prompt_attn_mask and attn_mask by padding with 0
        padding = torch.zeros(
            (attn_mask.shape[0], attn_mask.shape[1] -
             prompt_attn_mask.shape[1]),
        ).to(device)
        padded_prompt_attn_mask = torch.cat([prompt_attn_mask, padding], dim=1)
        prompt_mask = (padded_prompt_attn_mask == 1)

        # don't score prompt tokens
        attn_mask[prompt_mask] = 0

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels)
             * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

        # ? getting OOM error down the line
        del out_logits
        del perplexity_batch
        del encodings
        del prompt_encodings
        del padding

    torch.cuda.empty_cache()
    ppls = torch.tensor(ppls)
    print(f"Mean PPL: {ppls.mean()}")
    print(f"Max PPL: {ppls.max()}")
    print(f"Min PPL: {ppls.min()}")
    print(f"Std PPL: {ppls.std()}")
    return ppls


def compute_seq_rep_n(samples, tokenizer, n=3):
    """compute seq-rep-n metric"""
    n_gram_reps = []

    for s in samples:
        n_grams = []
        tokens = tokenizer(s, add_special_tokens=False).input_ids
        for i in range(len(tokens)):
            if i <= len(tokens) - n:
                n_grams.append(tuple(tokens[i:i + n]))

        rep = 1 - len(set(n_grams)) / len(n_grams)
        n_gram_reps.append(rep)

    median_rep = np.median(n_gram_reps)
    mean_rep = np.mean(n_gram_reps)
    return {
        f"median_seq_rep_{n}": median_rep,
        f"mean_seq_rep_{n}": mean_rep,
        f"list_seq_rep_{n}": n_gram_reps,
    }


def compute_total_rep_n(samples, tokenizer, n=3):
    """compute total-rep-n metric"""
    n_grams = []

    for s in samples:
        tokens = tokenizer(s, add_special_tokens=False).input_ids
        for i in range(len(tokens)):
            if i <= len(tokens) - n:
                n_grams.append(tuple(tokens[i:i + n]))

    total_rep = 1 - len(set(n_grams)) / len(n_grams)

    return {f"total_rep_{n}": total_rep}


def compute_repetition(data, tokenizer):
    """Compute repetition metrics."""
    samples = data["samples"]["model_text"]
    data.update(compute_seq_rep_n(samples, tokenizer, n=3))
    data.update(compute_total_rep_n(samples, tokenizer, n=3))
       
def initialize_oracle(name):
    # initialize oracle model
    oracle_tokenizer = AutoTokenizer.from_pretrained(
        name,
        device_map="auto")
    if oracle_tokenizer.pad_token is None:
        oracle_tokenizer.pad_token = oracle_tokenizer.eos_token

    oracle_model = AutoModelForCausalLM.from_pretrained(name,
                                                        device_map="auto", torch_dtype=torch.bfloat16)
    oracle_model.eval()
    return oracle_model, oracle_tokenizer


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_model", type=str,
                        help="Oracle model name", default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=8)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with open(args.output_file, 'r') as f:
        output_data = json.load(f)

    original_tokenizer = AutoTokenizer.from_pretrained(
        output_data["model_name"],
        device_map="auto")
    if original_tokenizer.pad_token is None:
        original_tokenizer.pad_token = original_tokenizer.eos_token

    oracle_model, oracle_tokenizer = initialize_oracle(args.oracle_model)
    ppls = compute_ppl(output_data["samples"], args.batch_size, "full_model_text",
                       oracle_model, oracle_tokenizer, original_tokenizer)
    # ppls = compute_ppl(dataset.to_dict(), args.batch_size, "full_human_text", oracle_model, oracle_tokenizer, original_tokenizer)

    # Add the PPL values to the output data
    ppl = {
        "mean": ppls.mean().item(),
        "max": ppls.max().item(),
        "min": ppls.min().item(),
        "std": ppls.std().item(),
        "median": ppls.median().item(),
        "oracle_model": args.oracle_model,
    }

    output_data["ppl"] = ppl


    # Compute repetition metrics
    compute_repetition(output_data, original_tokenizer)

    # Write the updated data back to the file
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
