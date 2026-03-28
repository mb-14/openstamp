#!/usr/bin/env python3
"""
Publish (save and optionally push) a watermarked model. Converted from notebooks/publish.ipynb.
"""

import argparse
import json
import os
import sys

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

LLAMA = "meta-llama/Llama-2-7b-hf"
MISTRAL = "mistralai/Mistral-7B-v0.3"
QWEN = "Qwen/Qwen2.5-7B"
PHI4 = "microsoft/phi-4"
SMOLLM2 = "HuggingFaceTB/SmolLM2-1.7B"
OLMO = "allenai/Olmo-3-1025-7B"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi

from src.openstamp import OpenStamp, Mode
from src.gaussmark import GaussMark


def get_param(model, name: str):
    mod = model
    parts = name.split(".")
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return getattr(mod, parts[-1])


def load_selector_config(selector_matrix_dir: str):
    """Load selector matrix path and config from directory. Returns (config_dict or None, matrix_path)."""
    dir_path = os.path.abspath(selector_matrix_dir)
    matrix_path = os.path.join(dir_path, "selector_matrix.pth")
    if not os.path.isfile(matrix_path):
        raise FileNotFoundError(f"Selector matrix not found: {matrix_path}")

    config = None
    for config_name in ("selector_metrics.json", "config.json"):
        config_path = os.path.join(dir_path, config_name)
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            break

    return config, matrix_path


def parse_args():
    parser = argparse.ArgumentParser(description="Save watermarked model (and optionally push to Hugging Face).")
    parser.add_argument(
        "--model-name",
        type=str,
        default=LLAMA,
        help="Base model name or path.",
        choices=[LLAMA, MISTRAL, QWEN, PHI4, SMOLLM2, OLMO],
    )
    parser.add_argument(
        "--watermark-type",
        type=str,
        default="openstamp",
        choices=["gaussmark", "openstamp"],
        help="Type of watermark: gaussmark or openstamp.",
    )
    parser.add_argument(
        "--selector-matrix-dir",
        type=str,
        default=None,
        help="Directory containing selector_matrix.pth and config (selector_metrics.json or config.json). Required for openstamp.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=15485863,
        help="Random seed for watermark.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "output", "watermarked_models"),
        help="Directory to save the watermarked model.",
    )
    parser.add_argument(
        "--publish-to-huggingface",
        action="store_true",
        default=False,
        help="If set, push the saved model to Hugging Face Hub.",
    )
    return parser.parse_args()


def model_slug(model_name: str) -> str:
    slug_map = {
        LLAMA: "llama2-7b",
        MISTRAL: "mistral-7b-v0.3",
        QWEN: "qwen2.5-7b",
        PHI4: "phi-4",
        SMOLLM2: "smollm2-1.7b",
        OLMO: "olmo-3-1025-7b",
    }
    if model_name in slug_map:
        return slug_map[model_name]
    return model_name.split("/")[-1].lower().replace(".", "-")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = args.model_name
    watermark_type = args.watermark_type
    seed = args.seed
    watermark_config = None

    print(f"Model name: {model_name}")
    print(f"Watermark type: {watermark_type}")
    print(f"Seed: {seed}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )

    if watermark_type == "gaussmark":
        if model_name not in (LLAMA, MISTRAL):
            raise ValueError("gaussmark is only supported for Llama-2-7B and Mistral-7B-v0.3")
        if model_name == LLAMA:
            target_layer = "model.layers.27.mlp.up_proj.weight"
            sigma = 0.04
        elif model_name == MISTRAL:
            target_layer = "model.layers.20.mlp.up_proj.weight"
            sigma = 0.005

        base_target_param = get_param(
            model, target_layer
        ).detach().clone().cpu()

        watermark = GaussMark(
            sigma=sigma,
            tokenizer=tokenizer,
            model=model,
            seed=seed,
            target_param_name=target_layer,
        )

        wm_target_param = get_param(
            watermark.model, target_layer
        ).detach().cpu()

        diff = wm_target_param - base_target_param
        l2 = diff.norm().item()
        print(f"L2 norm of watermark diff: {l2}")
        std = diff.std(unbiased=False).item()

        watermark_config = {
            "watermark_type": "gaussmark",
            "sigma": sigma,
            "seed": seed,
            "target_param_name": target_layer,
        }

    elif watermark_type == "openstamp":
        if args.selector_matrix_dir is None:
            raise ValueError("--selector-matrix-dir is required for watermark-type openstamp")
        config, matrix_path = load_selector_config(args.selector_matrix_dir)
        final_weight = torch.load(matrix_path, map_location="cpu")
        L = final_weight.shape[0]
        delta = 1.0
        gamma = 0.25
        sem_align = bool(config.get("sem_align", False)) if config else False
        align_method = config.get("align_method") if config else None
        embedding_model = config.get("embedding_model") if config else None

        # Reference to selector matrix path (relative to model save dir)
        selector_matrix_filename = "selector_matrix.pth"
        watermark_config = {
            "watermark_type": "openstamp",
            "delta": delta,
            "gamma": gamma,
            "seed": seed,
            "L": int(L),
            "selector_matrix_path": selector_matrix_filename,
        }

        print(f"L = {L}, sem_align = {sem_align}, align_method = {align_method}, embedding_model = {embedding_model}")

        base_target_param = get_param(model, "lm_head.weight").detach().clone().cpu()

        watermark = OpenStamp.from_config(
            delta=delta,
            gamma=gamma,
            seed=seed,
            final_weight=final_weight,
            model=model,
            tokenizer=tokenizer,
            unembedding_param_name="lm_head",
            mode=Mode.Generate,
        )

        wm_target_param = get_param(watermark.model, "lm_head.weight").detach().cpu()
        diff = wm_target_param - base_target_param
        l2 = diff.norm().item()
        print(f"L2 norm of watermark diff: {l2}")
        std = diff.std(unbiased=False).item()

    print(f"Standard deviation of watermark diff: {std}")

    watermarked_model = watermark.model

    model_name_slug = model_slug(model_name)
    if watermark_type == "gaussmark":
        watermarked_repo_name = f"{model_name_slug}-gaussmark-s{watermark.sigma}"
    else:
        watermarked_repo_name = f"{model_name_slug}-openstamp-L{L}-delta{delta}-gamma{gamma}"

    print(f"Watermarked model name: {watermarked_repo_name}")

    save_path = os.path.join(args.output_dir, watermarked_repo_name)
    os.makedirs(save_path, exist_ok=True)
    watermarked_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if watermark_config is not None:
        config_save_path = os.path.join(save_path, "watermark_config.json")
        with open(config_save_path, "w") as f:
            json.dump(watermark_config, f, indent=2)
        print(f"Watermark config saved to {config_save_path}")

    readme_template_path = os.path.join(REPO_ROOT, "assets", "README.md")
    if os.path.isfile(readme_template_path):
        with open(readme_template_path, "r") as f:
            readme_template = f.read()
        readme_data = dict(watermark_config) if watermark_config else {}
        readme_data["base_model"] = model_name
        config_json = json.dumps(readme_data, indent=2)
        readme_filled = readme_template.replace("{{CONFIG_JSON}}", config_json)
        readme_dest = os.path.join(save_path, "README.md")
        with open(readme_dest, "w") as f:
            f.write(readme_filled)
        print(f"README written to {readme_dest}")

    print(f"Watermarked model saved to {save_path}")

    if args.publish_to_huggingface:
        repo_id = f"openstamp/{watermarked_repo_name}"
        print(f"Pushing to Hugging Face Hub as {repo_id} ...")
        watermarked_model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        if watermark_config is not None:
            config_save_path = os.path.join(save_path, "watermark_config.json")
            HfApi().upload_file(
                path_or_fileobj=config_save_path,
                path_in_repo="watermark_config.json",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add watermark config",
            )
        # Upload selector_matrix.pth if present
        if args.selector_matrix_dir is not None:
            selector_matrix_path = os.path.join(os.path.abspath(args.selector_matrix_dir), "selector_matrix.pth")
            if os.path.isfile(selector_matrix_path):
                HfApi().upload_file(
                    path_or_fileobj=selector_matrix_path,
                    path_in_repo="selector_matrix.pth",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Add selector matrix",
                )
        readme_hub = os.path.join(save_path, "README.md")
        if os.path.isfile(readme_hub):
            HfApi().upload_file(
                path_or_fileobj=readme_hub,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add model README",
            )
        print("Done pushing to Hugging Face Hub.")


if __name__ == "__main__":
    main()
