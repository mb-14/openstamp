import os
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from rich import print as rprint
from typing import Dict, List
from tqdm import tqdm
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse


def load_decoder_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_dataloader(dataset_path: str, batch_size: int, total_samples: int):
    path = os.path.join(dataset_path, "prefixes.pt")
    tensor = torch.load(path)
    if total_samples > 0:
        tensor = tensor[:total_samples]
    data = TensorDataset(torch.load(path))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=True)
    return dataloader


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    # model type - llm or sentence transformer

    parser.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Huggingface model name")

    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer used to decode the dataset")

    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for data processing")
    
    parser.add_argument("--total_samples", type=int, default=-1,
                        help="Number of samples to process from the dataset")

    args = parser.parse_args()
    # Print the arguments
    rprint(f"Arguments: {args}")
    torch.manual_seed(42)

    decoder_tokenizer = load_decoder_tokenizer(args.tokenizer)

    sent_model = SentenceTransformer(
        args.model, device='cuda', trust_remote_code=True).bfloat16()
    sent_model.eval()

    model_name = args.model.split("/")[-1]

    dataloader = get_dataloader(args.dataset_path, args.batch_size, args.total_samples)

    all_embeddings = []
    pbar = tqdm(total=len(dataloader), desc="Processing batches")
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to("cuda")
            # Decode the input ids to text
            decoded_batch = decoder_tokenizer.batch_decode(
                batch, skip_special_tokens=True)
            # Encode the text using the sentence transformer model
            embeddings = sent_model.encode(
                decoded_batch, convert_to_tensor=True)
            all_embeddings.append(embeddings.detach().cpu())
            del batch
            del decoded_batch
            del embeddings
            torch.cuda.empty_cache()
            pbar.update(1)

    pbar.close()

    all_embeddings = torch.cat(all_embeddings, dim=0)

    save_dir = os.path.join(args.dataset_path, f"embeddings_{model_name}.pt")

    torch.save(all_embeddings, save_dir)
