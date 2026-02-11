import argparse
import json
import os
import time
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCTION = (
    "The input contains a Prompt and its Completion. Paraphrase ONLY the Completion. "
    "Preserve the exact meaning, facts, scope, and intent. "
    "Do not add, remove, infer, or reinterpret any information. "
    "Change wording and sentence structure only."
)


class LlamaParaphraser:
    def __init__(self, model_name: str, verbose: bool = True):
        time_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        if verbose:
            print(f"{model_name} loaded in {time.time() - time_start:.2f}s")

    def infer_max_new_tokens(self, completions: List[str], buffer: int = 50) -> int:
        if not completions:
            return buffer
        lengths = [len(self.tokenizer.encode(text)) for text in completions]
        return max(lengths) + buffer

    def _build_messages(self, prompt: str, completion: str) -> List[dict]:
        user_content = (
            f"Instruction: {INSTRUCTION}\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Completion:\n{completion}\n\n"
            f"Paraphrased Completion:"
        )
        return [
            {"role": "system", "content": "You are a precise paraphrasing assistant."},
            {"role": "user", "content": user_content},
        ]

    def paraphrase_batch(
        self,
        prompts: List[str],
        completions: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
    ) -> List[str]:
        conversations = [self._build_messages(p, c) for p, c in zip(prompts, completions)]
        input_texts = [
            self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=num_beams <= 1,
                temperature=temperature if num_beams <= 1 else None,
                top_p=top_p if num_beams <= 1 else None,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results: List[str] = []
        input_seq_len = inputs.input_ids.shape[1]
        for i in range(outputs.shape[0]):
            gen_ids = outputs[i, input_seq_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)
        return results


class ChatGPTParaphraser:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = True,
    ):
        time_start = time.time()
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ImportError(
                "OpenAI client not available. Install with: pip install openai"
            ) from exc

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key missing. Set --openai_api_key or OPENAI_API_KEY.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        if verbose:
            print(f"{model_name} (OpenAI API) ready in {time.time() - time_start:.2f}s")

    def paraphrase_one(
        self,
        prompt: str,
        completion: str,
        max_new_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        user_content = (
            f"Instruction: {INSTRUCTION}\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Completion:\n{completion}\n\n"
            f"Paraphrased Completion:"
        )
    
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": "You are a precise paraphrasing assistant."},
                {"role": "user", "content": user_content},
            ],
            max_output_tokens=max_new_tokens,  # Add buffer for reasoning tokens
            # reasoning={"effort": "low"}
        )
        return response.output_text


def chunked(iterable: List[str], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield i, iterable[i : i + batch_size]


def model_suffix_from_name(name: str) -> str:
    return (
        name.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def build_paraphraser(args):
    if args.provider == "openai":
        model_name = args.openai_model
        paraphraser = ChatGPTParaphraser(
            model_name=model_name,
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
        )
    else:
        model_name = "Qwen/Qwen2.5-14B-Instruct"
        paraphraser = LlamaParaphraser(model_name=model_name)
    return model_name, paraphraser


def run_paraphrasing(
    paraphraser,
    provider: str,
    prompts: List[str],
    completions: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
    batch_size: int,
) -> List[str]:
    paraphrases: List[str] = []
    if provider == "openai":
        for prompt, completion in tqdm(zip(prompts, completions), total=len(prompts)):
            out = paraphraser.paraphrase_one(
                prompt,
                completion,
                max_new_tokens=max_new_tokens,
                temperature=None,
                top_p=None,
            )
            paraphrases.append(out)
    else:
        for start_idx, batch_prompts in tqdm(
            chunked(prompts, batch_size),
            total=(len(prompts) + batch_size - 1) // batch_size,
        ):
            batch_completions = completions[start_idx : start_idx + batch_size]
            inferred_max_new_tokens = paraphraser.infer_max_new_tokens(batch_completions)
            batch_out = paraphraser.paraphrase_batch(
                batch_prompts,
                batch_completions,
                max_new_tokens=inferred_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
            )
            paraphrases.extend(batch_out)
    return paraphrases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--provider", type=str, choices=["hf", "openai"], default="hf")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--openai_model", type=str, default="gpt-4.1")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(42)

    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    model_name, paraphraser = build_paraphraser(args)
    model_suffix = model_suffix_from_name(model_name)

    with open(args.output_file, "r") as f:
        data = json.load(f)

    samples = data["samples"]
    prompts = samples["prompt_text"]
    completions = samples["model_text"]

    if args.limit is not None:
        prompts = prompts[: args.limit]
        completions = completions[: args.limit]

    paraphrases = run_paraphrasing(
        paraphraser,
        args.provider,
        prompts,
        completions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
    )

    samples["llm_paraphrase"] = paraphrases
    samples[f"{model_suffix}_paraphrase"] = paraphrases

    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
