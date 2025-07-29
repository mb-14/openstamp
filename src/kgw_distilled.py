from src.kgw.watermark_processor import WatermarkDetector
import torch


class KGWDistilled:
    def __init__(self, gamma, seeding_scheme, kgw_device, tokenizer, model=None):
        self.tokenizer = tokenizer
        if model is not None:
            self.model = model
            self.model.eval()
        self.detector = WatermarkDetector(
            device=kgw_device,
            tokenizer=tokenizer,
            vocab=tokenizer.get_vocab().values(),
            gamma=gamma,
            seeding_scheme=seeding_scheme,
            normalizers=[],
        )

    def score_text_batch(self, batch_text):
        all_scores = []
        for text in batch_text:
            score = self.detector.detect(text)
            z_score = score["z_score"]
            all_scores.append(z_score)
        all_scores = torch.tensor(all_scores, dtype=torch.float32)
        return all_scores
