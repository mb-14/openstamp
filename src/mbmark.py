import torch
from functools import cached_property
import scipy.stats
import numpy as np
from tqdm import tqdm
import torch.distributions as D
import math

# Create ENUM for the different modes


class Mode:
    Generate = "generate"
    Detect = "detect"


class HiddenStateExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = model.device
        self.model.eval()

    def forward(self, x):
        # Call the base model to get hidden states before the final projection
        outputs = self.model.model(**x, return_dict=True)
        # shape: [batch_size, seq_len, hidden_dim]
        return outputs.last_hidden_state


class MbMark:

    model_to_hs_norm = {
        "meta-llama/Llama-2-7b-hf": 118.0,
        "meta-llama/Llama-2-7b-chat-hf": 114.9,
        "meta-llama/Llama-3.1-8B": 142.88,
        "mistralai/Mistral-7B-v0.3": 365.22
    }

    model_to_var_factor = {
        "meta-llama/Llama-2-7b-hf": 74.32,
    }

    def __init__(self, model, tokenizer, unembedding_param_name, augmented_unembedding, mode, binom_detect=False, delta=None, gamma=None, final_weight=None, watermarking_matrix=None):
        self.model = model
        self.tokenizer = tokenizer
        self.unembedding = getattr(model, unembedding_param_name).weight.data
        self.augmented_unembedding = augmented_unembedding
        self.delta = delta
        self.gamma = gamma
        self.final_weight = final_weight
        self.binom_detect = binom_detect
        self.watermarking_matrix = watermarking_matrix

        if mode == Mode.Detect:
            self.cluster_detector = HiddenStateExtractor(
                model)
            self.cluster_detector.eval()
        elif mode == Mode.Generate:
            with torch.no_grad():
                getattr(model, unembedding_param_name).weight.data.copy_(
                    augmented_unembedding)

    @classmethod
    def sample_beta(cls, size, alpha, beta, seed, device='cpu'):
        """
        Samples from Beta(alpha, beta) using a fixed seed.
        Returns a tensor of shape `size`.
        """
        # Use reparameterization trick: sample two Gamma(α,1) variables and normalize
        rng = np.random.Generator(np.random.PCG64(seed=seed))
        samples_np = rng.beta(alpha, beta, size=size)
        samples = torch.from_numpy(samples_np).float()
        return samples.to(device)

    @classmethod
    def mb(cls, delta, gamma, seed, final_weight, model, tokenizer, unembedding_param_name, binom_detect=False, mode=Mode.Detect):
        unembedding = getattr(model, unembedding_param_name).weight.data
        vocab_size = len(tokenizer)

        assert unembedding.shape[0] == vocab_size, \
            "Unembedding matrix and tokenizer must have the same vocab size"

        assert final_weight.shape[1] == model.config.hidden_size, \
            "Final weight and model hidden size must match"

        watermark_matrix = cls._make_watermarking_matrix(
            vocab_size, delta, gamma, seed, final_weight.shape[0])
        delta_mat = (watermark_matrix @
                     final_weight).to(unembedding.device).to(unembedding.dtype)
        augmented_unembedding = unembedding.clone() + delta_mat

        return cls(model, tokenizer, unembedding_param_name, augmented_unembedding, mode, binom_detect, delta=delta, gamma=gamma, final_weight=final_weight, watermarking_matrix=watermark_matrix)

    @classmethod
    def noise_injection(cls, delta, seed, model, tokenizer, unembedding_param_name, distribution, mode=Mode.Detect):
        alpha = 0.5
        unembedding = getattr(model, unembedding_param_name).weight.data
        vocab_size = len(tokenizer)
        hs_norm = cls.model_to_hs_norm[model.config.name_or_path]
        hidden_size = unembedding.shape[1]

        if distribution == "symmetric_beta":
            delta_mat = cls.sample_beta(
                size=(vocab_size, hidden_size), alpha=alpha, beta=alpha, seed=seed, device=unembedding.device)
            delta_mat = delta_mat * 2 - 1  # Scale to [-1, 1]
            variance_scale = math.sqrt(2*alpha + 1)
        elif distribution == "gaussian":
            delta_mat = torch.randn(
                size=(vocab_size, hidden_size), generator=torch.Generator().manual_seed(seed))
            variance_scale = 1
        elif distribution == "uniform":
            delta_mat = torch.rand(
                size=(vocab_size, hidden_size), generator=torch.Generator().manual_seed(seed))
            delta_mat = delta_mat * 2 - 1  # Scale to [-1, 1]
            variance_scale = math.sqrt(3)
        elif distribution == "truncated_normal":
            generator = torch.Generator().manual_seed(seed)
            # Compute bounds in standard-normal space
            alpha_std = -2.0
            beta_std = 2.0
            # Convert bounds to CDF space

            def standard_normal_cdf(x): return 0.5 * \
                (1 + torch.erf(x / math.sqrt(2)))
            phi_alpha = standard_normal_cdf(torch.tensor(alpha_std))
            phi_beta = standard_normal_cdf(torch.tensor(beta_std))
            # Sample uniformly from [CDF(a), CDF(b)]
            u = torch.rand((vocab_size, hidden_size), generator=generator)
            u = phi_alpha + u * (phi_beta - phi_alpha)
            # Inverse CDF using erfinv
            z = math.sqrt(2) * torch.erfinv(2 * u - 1)
            delta_mat = z
            variance_scale = 1.0  # already standardized
        elif distribution == "low_rank":
            rank = min(32, hidden_size)                       # configurable r
            g = torch.Generator().manual_seed(seed)
            U = torch.randn((vocab_size, rank), generator=g)
            V = torch.randn((hidden_size, rank), generator=g)
            delta_mat = (U @ V.T)          # unit variance
            variance_scale = 1 / math.sqrt(rank)  # scale by sqrt(rank)

        delta_mat = delta_mat * delta * variance_scale / hs_norm
        delta_mat = delta_mat.to(unembedding.device).to(unembedding.dtype)

        augmented_unembedding = unembedding.clone() + delta_mat
        return cls(model, tokenizer, unembedding_param_name, augmented_unembedding, mode, delta=delta)

    @classmethod
    def from_augmented(cls, augmented_unembedding, model, tokenizer, unembedding_param_name, mode=Mode.Detect):
        assert augmented_unembedding.shape[0] == len(tokenizer), \
            "Augmented unembedding matrix and tokenizer must have the same vocab size"
        assert augmented_unembedding.shape[1] == model.config.hidden_size, \
            "Augmented unembedding matrix and model hidden size must match"
        return cls(model, tokenizer, unembedding_param_name, augmented_unembedding, mode)

    @staticmethod
    def _make_watermarking_matrix(vocab_size, delta, gamma, seed, n_clusters):
        def prf_lookup(cluster, seed):
            return seed * cluster

        def get_partition(cluster):
            rng = torch.Generator()
            prf_key = prf_lookup(cluster, seed)
            rng.manual_seed(prf_key % (2**64 - 1))

            greenlist_size = int(vocab_size * gamma)
            vocab_permutation = torch.randperm(vocab_size, generator=rng)
            greenlist_ids = vocab_permutation[:greenlist_size]
            redlist_ids = vocab_permutation[greenlist_size:]
            return greenlist_ids, redlist_ids

        matrix = torch.zeros(vocab_size, n_clusters)
        for i in range(n_clusters):
            greenlist_ids, _ = get_partition(i)
            matrix[greenlist_ids, i] = delta
        return matrix

    @torch.no_grad()
    def llr_raw(self, hidden_states, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        labels = input_ids[:, 2:].to(self.unembedding.device)
        attention_mask = attention_mask[:, 2:].to(
            self.unembedding.device)
        hidden_states = hidden_states[:, 1:-1].to(self.unembedding.device)

        logits_base = hidden_states @ self.unembedding.T
        # (B, T, V)
        logits_marked = hidden_states @ self.augmented_unembedding.T

        # Compute log-probs
        log_probs_base = torch.nn.functional.log_softmax(
            logits_base, dim=-1)
        log_probs_marked = torch.nn.functional.log_softmax(
            logits_marked, dim=-1)

        # Gather the log-prob of the actual token
        log_probs_base = log_probs_base.gather(
            2, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
        log_probs_marked = log_probs_marked.gather(
            2, labels.unsqueeze(-1)).squeeze(-1)

        mask = attention_mask.bool()
        log_probs_base = log_probs_base.masked_fill(~mask, 0.0)
        log_probs_marked = log_probs_marked.masked_fill(~mask, 0.0)

        sum_ll_base = log_probs_base.sum(dim=1)
        sum_ll_marked = log_probs_marked.sum(dim=1)

        llr = sum_ll_marked - sum_ll_base

        llr = llr / attention_mask.sum(dim=1).float()

        # Clean up
        del inputs, hidden_states, log_probs_base, log_probs_marked
        del logits_base, logits_marked, labels
        del attention_mask, mask
        torch.cuda.empty_cache()
        return llr.cpu().float()

    @torch.no_grad()
    def ll(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(
            self.cluster_detector.device)
        hidden_states = self.cluster_detector(inputs)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        labels = input_ids[:, 2:].to(self.unembedding.device)
        attention_mask = attention_mask[:, 2:].to(self.unembedding.device)
        hidden_states = hidden_states[:, 1:-1].to(self.unembedding.device)

        logits = hidden_states @ self.augmented_unembedding.T
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        mask = attention_mask.bool()
        log_probs = log_probs.masked_fill(~mask, 0.0)
        sum_log_probs = log_probs.sum(dim=1)
        return sum_log_probs.cpu().float()

    @torch.no_grad()
    def grad_delta(self, text):
        # Recover N₀ from the current state
        N0 = (self.augmented_unembedding - self.unembedding) / self.delta

        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(
            self.cluster_detector.device
        )
        hidden_states = self.cluster_detector(inputs)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        labels = input_ids[:, 2:].to(self.unembedding.device)
        mask = attention_mask[:, 2:].to(self.unembedding.device).bool()
        hidden_states = hidden_states[:, 1:-1].to(self.unembedding.device)

        def grad_at(delta):
            W = self.unembedding + delta * N0
            logits = hidden_states @ W.T  # (B, T, V)
            probs = logits.softmax(dim=-1)
            exp_N = torch.einsum("btv,vd->btd", probs,
                                 N0)
            N_gold = N0[labels]
            grad_tok = (hidden_states * (N_gold - exp_N)
                        ).sum(-1)
            grad_tok = grad_tok.masked_fill(~mask, 0.0)

            grad_seq = grad_tok.sum(dim=1) / attention_mask.sum(dim=1).float()
            return grad_seq

        grad_0 = grad_at(0)
        grad_delta = grad_at(self.delta)

        score = grad_delta * 0.5 + grad_0

        return score.cpu().float()

    @torch.no_grad()
    def hessian_delta(self, text):
        """
        Estimate ∂²L/∂δ² using finite differences.
        Assumes:
            - self.unembedding is the base embedding (frozen)
            - self.augmented_unembedding = unembedding + δ₀ * N_dir
            - N0 = δ₀ * N_dir is already computed

        Args:
            text: list of strings (batch)

        Returns:
            Tensor of shape (batch,) — estimated second derivative
        """
        deltas = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32,
                              device=self.unembedding.device)

        N0 = self.augmented_unembedding - self.unembedding

        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(
            self.cluster_detector.device
        )
        hidden_states = self.cluster_detector(inputs)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        labels = input_ids[:, 2:].to(self.unembedding.device)
        mask = attention_mask[:, 2:].to(self.unembedding.device).bool()
        hidden_states = hidden_states[:, 1:-1].to(self.unembedding.device)

        # Prepare to collect loss at each delta
        losses = []

        for delta in deltas:
            W_delta = self.unembedding + delta * N0
            logits = hidden_states @ W_delta.T
            log_probs = logits.log_softmax(dim=-1)

            # log_probs: (B, T, V), labels: (B, T) → select log-probs of correct tokens
            tok_log_probs = log_probs.gather(
                2, labels.unsqueeze(-1)).squeeze(-1)

            # Zero out pad tokens
            tok_log_probs = tok_log_probs.masked_fill(~mask, 0.0)

            seq_log_prob = tok_log_probs.sum(
                dim=1) / attention_mask.sum(dim=1).float()
            loss = -seq_log_prob  # Negative log likelihood
            losses.append(loss)

        # Finite difference second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x−h)) / h²
        L_0 = losses[1]
        L_m = losses[0]
        L_p = losses[2]

        h = 0.1  # step size (delta offset from 1.0)

        hessian = (L_p - 2 * L_0 + L_m) / (h ** 2)
        return hessian.cpu().float()

    def binomial_count(self, hidden_states, inputs):
        batch_size = inputs.input_ids.shape[0]
        hidden_states = hidden_states.to(
            self.final_weight.device).to(self.final_weight.dtype)
        selectors = torch.matmul(hidden_states, self.final_weight.T)
        clusters = torch.argmax(selectors, dim=-1)
        vocab_size = self.watermarking_matrix.shape[0]

        # Remove BOS token
        clusters = clusters[:, 1:]
        # Remove BOS + shift by 1
        sequences = inputs.input_ids[:, 2:]

        counts = torch.zeros(batch_size, dtype=torch.int32)
        lengths = torch.zeros(batch_size, dtype=torch.int32)

        for i in range(batch_size):
            count = 0
            sequence = sequences[i]
            # Remove padding
            sequence = sequence[sequence != self.tokenizer.pad_token_id]
            n_tokens = sequence.shape[0]
            for j in range(n_tokens):
                cluster = clusters[i, j].item()

                watermarking_list = self.watermarking_matrix[:, cluster].bool(
                )
                # Get indices of true values
                green_list = torch.arange(vocab_size)[
                    watermarking_list]
                token = sequence[j].item()
                if token in green_list:
                    count += 1
            counts[i] = count
            lengths[i] = n_tokens

        del clusters, inputs
        del sequences

        torch.cuda.empty_cache()

        z = (counts - self.gamma * lengths) / \
            torch.sqrt(self.gamma * lengths * (1 - self.gamma))
        return z

    def score_text_batch(self, batch_text):
        # grad_scores = self.grad_delta(batch_text)
        # return grad_scores
        """
        Score a batch of text using the model and tokenizer.
        Args:
            batch_text: List of text samples to score.
            tokenizer: The tokenizer to use for scoring.
        """
        with torch.no_grad():
            inputs = self.tokenizer(batch_text, padding=True,
                                    return_tensors="pt").to(self.cluster_detector.device)
            hidden_states = self.cluster_detector(inputs)
            if self.binom_detect:
                llr_scores = self.binomial_count(hidden_states, inputs)
            else:
                llr_scores = self.llr_raw(hidden_states, inputs)

        return llr_scores
