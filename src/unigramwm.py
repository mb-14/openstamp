from src.unigram.gptwm import GPTWatermarkLogitsProcessor, GPTWatermarkDetector
import torch


class Unigram:
    def __init__(self, gamma, delta, hash_key, tokenizer, model=None):
        self.model = model
        self.tokenizer = tokenizer
        if self.model is not None:
            self.model.eval()
        self.detector = GPTWatermarkDetector(fraction=gamma,
                                             strength=delta,
                                             vocab_size=len(tokenizer),
                                             watermark_key=hash_key)
        self.watermark = GPTWatermarkLogitsProcessor(
            fraction=gamma,
            strength=delta,
            vocab_size=len(tokenizer),
            watermark_key=hash_key)

    def score_text_batch(self, batch_text):
        all_scores = []
        for text in batch_text:
            tokens = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
            z_score = self.detector.detect(tokens.tolist())
            all_scores.append(z_score)
        all_scores = torch.tensor(all_scores, dtype=torch.float32)
        return all_scores
