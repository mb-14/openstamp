from tqdm import tqdm
from datasets import load_from_disk
import torch
import argparse
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from nltk.tokenize import sent_tokenize
import re
import json


def preprocess_text(text):
    # Replace dot-dot-dot with ellipses
    text = re.sub(r"\.\s*\.\s*\.", "...", text)

    # Remove lingering single dots with no alphabet before/after
    text = re.sub(r"\s*\.\s*", " ", text)

    # Collapse multiple ellipses into one
    text = re.sub(r"\.{4,}", "...", text)

    return text


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(
            model, device_map="auto")
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=2, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [
            0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [
            0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        if len(sentences) > 40:
            print(f"Potentially buggy input text: {input_text}", flush=True)
            input_text = preprocess_text(input_text)
            print(f"Preprocessed to: {input_text}", flush=True)
            sentences = sent_tokenize(input_text)

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(
                sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer(
                [final_input_text], return_tensors="pt")
            final_input = {k: v.to(self.model.device)
                           for k, v in final_input.items()}

            with torch.no_grad():
                model_outputs = self.model.generate(**final_input, **kwargs)
                outputs = self.tokenizer.batch_decode(
                    model_outputs, skip_special_tokens=True)
                del final_input, model_outputs
                torch.cuda.empty_cache()
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lex", type=int, default=40,
                        help="Lexical diversity")
    parser.add_argument("--order", type=int, default=40,
                        help="Order diversity")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_file", type=str)

    args = parser.parse_args()

    torch.manual_seed(42)
    nltk.download('punkt_tab')

    # Print arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    dp = DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")

    # Load output_file JSON
    with open(args.output_file, "r") as f:
        data = json.load(f)

    samples = data["samples"]

    prompts = samples["prompt_text"]
    completions = samples["model_text"]
    paraphrases = []
    for (prompt, completion) in tqdm(zip(prompts, completions)):
        paraphrase = dp.paraphrase(input_text=completion, prefix=prompt, lex_diversity=args.lex, order_diversity=args.order,
                                   do_sample=True, top_p=0.75, top_k=None, max_length=args.max_length)
        paraphrases.append(paraphrase)

    column_name = f"dipper_text_lex{args.lex}_order{args.order}"

    samples[column_name] = paraphrases

    # Save updated data
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)
