import torch

from src.llr import length_normalized_llr


class GaussMark:
    def __init__(self, sigma, seed, target_param_name, tokenizer, model, llr_detection=False):
        """
        GaussMark-style structural watermarking for LLMs.
        Args:
            sigma: stddev of Gaussian perturbation.
            seed: seed for deterministic watermark key.
            target_param_name: string name of the layer to watermark (e.g. 'lm_head').
            tokenizer: HuggingFace tokenizer.
            model: the unwatermarked base model.
            llr_detection: if True, keep the base checkpoint clean and score with
                length-normalized LLR (two forwards: base vs key-augmented).
        """
        self.sigma = sigma
        self.seed = seed
        self.target_param_name = target_param_name
        self.tokenizer = tokenizer
        self.llr_detection = bool(llr_detection)
        self.model = model
        if not self.llr_detection:
            self.watermark_model(model)

    def watermark_key(self, shape):
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        return torch.randn(shape, generator=rng) * self.sigma

    def _target_weight(self):
        module = self.model
        for name in self.target_param_name.split('.')[:-1]:
            module = getattr(module, name)
        weight_name = self.target_param_name.split('.')[-1]
        return getattr(module, weight_name)

    def _add_key(self, sign=1.0):
        weight = self._target_weight()
        key = self.watermark_key(weight.shape).to(weight.device, weight.dtype)
        with torch.no_grad():
            weight.data.add_(sign * key)
        return weight

    def watermark_model(self, model):
        """
        Add Gaussian watermark to the model’s specified layer.
        """
        self.model = model
        self._add_key(sign=1.0)

    def score_text_batch(self, batch_text):
        if self.llr_detection:
            return self.llr_detect(batch_text)

        self.model.eval()

        with torch.enable_grad():
            # Tokenize full batch
            inputs = self.tokenizer(
                batch_text, padding=True, return_tensors="pt"
            ).to(self.model.device)
            input_ids = inputs.input_ids  # (B, T)
            attention_mask = inputs.attention_mask  # (B, T)

            # Forward pass
            logits = self.model(**inputs).logits  # (B, T, V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Next-token predictions
            shifted_input_ids = input_ids[:, 2:]              # (B, T-1)
            shifted_log_probs = log_probs[:, 1:-1, :]          # (B, T-1, V)
            log_probs_seq = torch.gather(
                shifted_log_probs, 2, shifted_input_ids.unsqueeze(-1)
            ).squeeze(-1)                                     # (B, T-1)

            # Build token mask (BOS and padding tokens ignored)
            shifted_attention_mask = attention_mask[:, 2:]    # (B, T-1)
            token_mask = (shifted_attention_mask == 1)        # (B, T-1)

            # Set log probs of masked-out tokens to 0
            log_probs_seq = log_probs_seq * token_mask

            # Compute log-likelihoods for each sequence
            log_likelihoods = log_probs_seq.sum(dim=-1)       # (B,)

            weight = self._target_weight()

            grads = []
            for i in range(log_likelihoods.shape[0]):
                # Compute per-sample gradient with shared computation graph
                grad_i = torch.autograd.grad(
                    outputs=log_likelihoods[i],
                    inputs=weight,
                    retain_graph=True,
                    create_graph=False,
                    only_inputs=True,
                    allow_unused=True
                )[0]

                if grad_i is None:
                    raise RuntimeError(f"Gradient is None for sample {i}.")
                grads.append(grad_i.view(-1).detach())

            grads = torch.stack(grads, dim=0)  # (B, D)
            keys = self.watermark_key(weight.shape).view(
                1, -1).to(grads.device)  # (1, D)

            # Compute test statistics
            dots = (keys * grads).sum(dim=1)              # (B,)
            norms = grads.norm(dim=1)                     # (B,)
            psi = dots / (self.sigma * norms)             # (B,)
            z_scores = psi

            self.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            return z_scores

    @torch.no_grad()
    def llr_detect(self, texts):
        """Length-normalized LLR under key-augmented vs base next-token dists."""
        if self.model is None:
            raise RuntimeError("LLR detection requires a base model")

        self.model.eval()
        device = next(self.model.parameters()).device
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        logits_base = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        self._add_key(sign=1.0)
        try:
            logits_marked = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        finally:
            self._add_key(sign=-1.0)

        return length_normalized_llr(logits_base, logits_marked, input_ids, attention_mask)
