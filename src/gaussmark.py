import torch
from functools import cached_property
import scipy.stats


class GaussMark:
    def __init__(self, sigma, seed, target_param_name, tokenizer, model):
        """
        GaussMark-style structural watermarking for LLMs.
        Args:
            sigma: stddev of Gaussian perturbation.
            seed: seed for deterministic watermark key.
            target_param_name: string name of the layer to watermark (e.g. 'lm_head').
            tokenizer: HuggingFace tokenizer.
            base_model: the unwatermarked base model for detection.
        """
        self.sigma = sigma
        self.seed = seed
        self.target_param_name = target_param_name
        self.tokenizer = tokenizer
        self.watermark_model(model)

    def watermark_key(self, shape):
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        return torch.randn(shape, generator=rng) * self.sigma

    def watermark_model(self, model):
        """
        Add Gaussian watermark to the model’s specified layer.
        """
        module = model
        for name in self.target_param_name.split('.')[:-1]:
            module = getattr(module, name)
        weight_name = self.target_param_name.split('.')[-1]
        weight = getattr(module, weight_name)

        with torch.no_grad():
            watermarked = weight.data + \
                self.watermark_key(weight.shape).to(
                    weight.device, weight.dtype)
            setattr(module, weight_name, torch.nn.Parameter(watermarked))

        self.model = model

    def score_text_batch(self, batch_text):
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

            # Locate the parameter to extract gradients from
            module = self.model
            for name in self.target_param_name.split('.')[:-1]:
                module = getattr(module, name)
            weight_name = self.target_param_name.split('.')[-1]
            weight = getattr(module, weight_name)

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
