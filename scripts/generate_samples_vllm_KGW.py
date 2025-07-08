import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TOKENIZERS_PARALLELISM'] = "false"
sys.path.append("/os-watermarking") #! add path for the watermarks
sys.path.append("/os-watermarking/MarkLLM") #! add path for the watermarks

from MarkLLM.watermark.auto_watermark import AutoWatermarkForVLLM
from MarkLLM.utils.transformers_config import TransformersConfig

import time
import json
import torch
import argparse

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed


def filter_length(example, min_length, tokenizer):
    return len(tokenizer(example[args.data_field], truncation=True, max_length=min_length)["input_ids"]) >= min_length


def encode(examples, min_length, tokenizer, device):
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


def evaluate_model(args):

    #* Check if the output file already exists
    if os.path.exists(args.output_file):
        print(f"File '{args.output_file}' already exists. Exiting.")
        sys.exit(0)  # Or use sys.exit(1) for an error code
        # with open(args.output_file, "r") as f:
        #     output_data = json.load(f)
    else:
        output_data = {}

    set_seed(args.seed)

    #* Load the model through VLLM
    model = LLM(args.model_name, tokenizer=args.model_name, 
                gpu_memory_utilization=0.9, 
                tensor_parallel_size=args.num_gpu, 
                dtype="bfloat16",
                tokenizer_mode="mistral" if "mistralai" in args.model_name.lower() else "auto"
            ) 
    config = AutoConfig.from_pretrained(args.model_name)

    transformers_config = TransformersConfig(
        model=AutoModelForCausalLM.from_pretrained(args.model_name),
        tokenizer=AutoTokenizer.from_pretrained(args.model_name),
        vocab_size=config.vocab_size,
        device="cuda",
        max_new_tokens=args.max_tokens
    )
    
    #* Load the relevant watermark
    wtm_config = f'config/{args.watermark}/prefix_{args.prefix_length}_gamma_{args.gamma}_delta_{args.delta}_key_{args.hash_key}.json'
    watermark = AutoWatermarkForVLLM(algorithm_name="KGW", 
                            algorithm_config=wtm_config, 
                            transformers_config=transformers_config)

    #* Load the prompts
    prompt_dataset = load_dataset(args.dataset_path, args.dataset_config_name, split=args.dataset_split, trust_remote_code=True, streaming=args.streaming)
    if args.dataset_path == "kmfoda/booksum":
     # Remove all columns except for the data_field
        prompt_dataset = prompt_dataset.remove_columns([col for col in prompt_dataset.column_names if col != args.data_field])

    #* Shuffle the dataset with a fixed seed
    prompt_dataset = prompt_dataset.shuffle(seed=args.seed)
    min_length = args.prompt_length + args.max_tokens

    #* Filter the encode the dataset
    # tokenizer = model.llm_engine.tokenizer.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_dataset = prompt_dataset.select(range(min(10000, len(prompt_dataset))))
    prompt_dataset = prompt_dataset.filter(filter_length, fn_kwargs={"min_length": min_length, "tokenizer": tokenizer})
    prompt_dataset = prompt_dataset.map(encode, fn_kwargs={"min_length": min_length, "tokenizer": tokenizer, "device": "cuda"}, batched=True)

    # test_prompts = []
    full_human_text = []
    prompt_text = []
    human_text = []
    for i, example in enumerate(prompt_dataset):
        full_human_text.append( example['text'] )
        prompt_text.append( example['prompt_text'] )
        human_text.append(  example['text_completion'] )

        if i == args.num_samples-1:
            break
    
    #* Let's actually sameple the responses
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        n=1,
        logits_processors=[watermark]
    )
    sampling_params.stop_token_ids = [model.llm_engine.tokenizer.tokenizer.eos_token_id]
    print("Generating test outputs...")
    print(prompt_text[0])

    #* Sample from the model
    start_time = time.time()
    outputs = model.generate(prompts=prompt_text, sampling_params=sampling_params, use_tqdm=True)
    end_time = time.time()
    model_text = []
    full_model_text = []
    for input, output in zip(prompt_text, outputs):
        model_text.append( output.outputs[0].text )
        full_model_text.append( input + output.outputs[0].text )

    #* Create dict
    data = {
        "prompt_text": prompt_text,
        "human_text": human_text,
        "full_human_text": full_human_text,
        "model_text": model_text,
        "full_model_text": full_model_text
    }


    #* Create dict of the watemarking hyperparams
    config = {
        "watermark": args.watermark,
        "gamma": args.gamma,
        "delta": args.delta,
        "hash_key": args.hash_key,
        "prefix_length": args.prefix_length
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
    
    time_taken = end_time - start_time
    print("Time taken:", time_taken)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #* Fixed defaults
    parser.add_argument('--prompt_length', type=int, default=50)
    parser.add_argument('--max_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--multinomial', action='store_true', default=True)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument("--num_gpu", type=int, default=1)

    #* watermark hyperparameters
    parser.add_argument('--watermark', type=str, default="KGW")
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--hash_key', type=int, default=15485863, help="PRF for the watermarking matrix")
    parser.add_argument('--prefix_length', type=int, default=1) #! the context length used to seed the vocab split

    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--seed', type=int, default=42)


    #* dataset used for collecting the prompts
    parser.add_argument('--dataset_path', type=str, default="allenai/c4")
    parser.add_argument('--dataset_config_name', type=str, default=None)
    parser.add_argument('--dataset_split', type=str, default="validation")
    parser.add_argument('--data_field', type=str, default="text")
    parser.add_argument("--streaming", action="store_true", default=False)

    args = parser.parse_args()

    #! Just unwatermarking for now
    #* SAVE JSON Config for the watermark file
    wtm_config_data = {
        "algorithm_name": args.watermark,
        "hash_key": args.hash_key,
        "prefix_length": args.prefix_length,
        "z_threshold": 4.0,
        "f_scheme": "time",
        "window_scheme": "left",
        "delta": args.delta,
        "gamma": args.gamma
    }
    folder = f"config/{args.watermark}"
    # os.makedirs
    filename = f"prefix_{args.prefix_length}_gamma_{args.gamma}_delta_{args.delta}_key_{args.hash_key}.json"
    with open( f"{folder}/{filename}" , 'w') as json_file:
        json.dump(wtm_config_data, json_file, indent=4)

    print(args.output_file)

    evaluate_model(args)
    
    torch.cuda.empty_cache()  # Frees unused memory
    torch.cuda.synchronize()   # Ensures all computations are finished