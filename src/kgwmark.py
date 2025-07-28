from src.kgw.watermark_processor import WatermarkDetector, WatermarkLogitsProcessor
import torch


class KGWMark:
    def __init__(self, model, gamma, delta, hash_key, kgw_device, tokenizer, llr_detection=False):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.detector = WatermarkDetector(
            device=kgw_device,
            tokenizer=tokenizer,
            vocab=tokenizer.get_vocab().values(),
            gamma=gamma,
            seeding_scheme="simple_1",
            normalizers=[],
            hash_key=hash_key,
        )
        self.watermark = WatermarkLogitsProcessor(
            vocab=tokenizer.get_vocab().values(),
            gamma=gamma,
            delta=delta,
            seeding_scheme="simple_1",
            hash_key=hash_key,
            device=kgw_device)
        self.llr_detection = llr_detection

    def score_text_batch(self, batch_text):
        if self.llr_detection:
            return self.llr_detect(batch_text)
        all_scores = []
        for text in batch_text:
            score = self.detector.detect(text)
            z_score = score["z_score"]
            all_scores.append(z_score)
        all_scores = torch.tensor(all_scores, dtype=torch.float32)
        return all_scores

    @torch.no_grad()
    def llr_detect(self, texts):
        # Tokenize input batch
        encodings = self.tokenizer(
            texts, return_tensors="pt", padding=True).to(self.model.device)
        input_ids = encodings.input_ids  # (B, T)
        attention_mask = encodings.attention_mask  # (B, T)

        # Get model logits
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        # Base log-probs
        log_probs_base = torch.nn.functional.log_softmax(logits, dim=-1)

        # Clone logits and apply watermark processor token-by-token
        logits_marked = logits.clone()
        B, T, V = logits.shape
        for b in range(B):
            for t in range(1, T):  # start at 1 to skip BOS token
                if attention_mask[b, t] == 0:
                    continue
                prefix = input_ids[b, :t+1].unsqueeze(0)         # (1, t+1)
                raw_scores = logits_marked[b, t].unsqueeze(0)    # (1, V)
                logits_marked[b, t] = self.watermark(
                    input_ids=prefix, scores=raw_scores).squeeze(0)

        # Log probs after watermarking
        log_probs_marked = torch.nn.functional.log_softmax(
            logits_marked, dim=-1)
        # Shift for next-token prediction
        labels = input_ids[:, 2:]
        log_probs_base = log_probs_base[:, 1:-1, :]
        log_probs_marked = log_probs_marked[:, 1:-1, :]
        attention_mask = attention_mask[:, 2:]

        # Gather correct token log-probs
        log_probs_base = log_probs_base.gather(
            2, labels.unsqueeze(-1)).squeeze(-1)
        log_probs_marked = log_probs_marked.gather(
            2, labels.unsqueeze(-1)).squeeze(-1)

        # Zero out padding via mask
        mask = attention_mask.bool()
        log_probs_base = log_probs_base.masked_fill(~mask, 0.0)
        log_probs_marked = log_probs_marked.masked_fill(~mask, 0.0)

        # Compute LLR per sample
        ll_base = log_probs_base.sum(dim=1)
        ll_marked = log_probs_marked.sum(dim=1)
        llr = (ll_marked - ll_base) / mask.sum(dim=1).float()

        return llr.cpu()
