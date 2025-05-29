import torch
from functools import cached_property
import scipy.stats
import numpy as np

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
        "mistralai/Mistral-7B-v0.3": 365.22
    }

    def __init__(self, model, tokenizer, unembedding_param_name, augmented_unembedding, mode):
        self.model = model
        self.tokenizer = tokenizer
        self.unembedding = getattr(model, unembedding_param_name).weight.data
        self.augmented_unembedding = augmented_unembedding

        if mode == Mode.Detect:
            self.cluster_detector = HiddenStateExtractor(
                model)
            self.cluster_detector.eval()
        elif mode == Mode.Generate:
            with torch.no_grad():
                getattr(model, unembedding_param_name).weight.data.copy_(
                    augmented_unembedding)

    @classmethod
    def sample_beta(cls, size, beta, seed, device='cpu'):
        """
        Samples from Beta(alpha, beta) using a fixed seed.
        Returns a tensor of shape `size`.
        """
        # Use reparameterization trick: sample two Gamma(α,1) variables and normalize
        rng = np.random.Generator(np.random.PCG64(seed=seed))
        samples_np = rng.beta(beta, beta, size=size)
        samples = torch.from_numpy(samples_np).float()
        samples = samples * 2 - 1
        return samples.to(device)

    @classmethod
    def mb(cls, delta, gamma, seed, final_weight, model, tokenizer, unembedding_param_name, mode=Mode.Detect):
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

        return cls(model, tokenizer, unembedding_param_name, augmented_unembedding, mode)

    @classmethod
    def mb2(cls, delta, seed, model, tokenizer, unembedding_param_name, mode=Mode.Detect):
        unembedding = getattr(model, unembedding_param_name).weight.data
        vocab_size = len(tokenizer)
        hs_norm = cls.model_to_hs_norm[model.config.name_or_path]
        hidden_size = unembedding.shape[1]

        delta_mat = cls.sample_beta(
            size=(vocab_size, hidden_size), beta=0.5, seed=seed, device=unembedding.device)
        delta_mat = delta_mat * delta / hs_norm
        delta_mat = delta_mat.to(unembedding.device).to(unembedding.dtype)

        augmented_unembedding = unembedding.clone() + delta_mat
        return cls(model, tokenizer, unembedding_param_name, augmented_unembedding, mode)

    @classmethod
    def mb3(cls, delta, seed, model, tokenizer, unembedding_param_name, mode=Mode.Detect):
        unembedding = getattr(model, unembedding_param_name).weight.data
        vocab_size = len(tokenizer)
        hs_norm = cls.model_to_hs_norm[model.config.name_or_path]
        hidden_size = unembedding.shape[1]

        rng = torch.Generator()
        rng.manual_seed(seed)
        U, _ = torch.linalg.qr(torch.randn(
            vocab_size, hidden_size, generator=rng))  # tall matrix
        # Step 2: V ∈ ℝ^{d × d}, orthonormal square
        V, _ = torch.linalg.qr(torch.randn(
            hidden_size, hidden_size, generator=rng))  # square matrix
        delta_mat = (U @ V.T)  # Shape: (V, d), full-rank, spectrally flat
        delta_mat = delta_mat * delta * (len(tokenizer)**0.5) / hs_norm
        delta_mat = delta_mat.to(unembedding.device).to(unembedding.dtype)

        augmented_unembedding = unembedding.clone() + delta_mat
        return cls(model, tokenizer, unembedding_param_name, augmented_unembedding, mode)

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
        del sum_ll_base, sum_ll_marked
        torch.cuda.empty_cache()
        return llr.cpu().float()

    # def binomial_count(self, hidden_states, inputs):
    #     batch_size = inputs.input_ids.shape[0]
    #     selectors = torch.matmul(hidden_states.to(
    #         self.final_weight.dtype), self.final_weight.T)
    #     clusters = torch.argmax(selectors, dim=-1)

    #     # Remove BOS token
    #     clusters = clusters[:, 1:]
    #     # Remove BOS + shift by 1
    #     sequences = inputs.input_ids[:, 2:]

    #     counts = torch.zeros(batch_size, dtype=torch.int32)
    #     lengths = torch.zeros(batch_size, dtype=torch.int32)

    #     for i in range(batch_size):
    #         count = 0
    #         sequence = sequences[i]
    #         # Remove padding
    #         sequence = sequence[sequence != self.tokenizer.pad_token_id]
    #         n_tokens = sequence.shape[0]
    #         for j in range(n_tokens):
    #             cluster = clusters[i, j].item()

    #             watermarking_list = self.watermarking_matrix[:, cluster].bool(
    #             )
    #             # Get indices of true values
    #             green_list = torch.arange(self.vocab_size)[
    #                 watermarking_list]
    #             token = sequence[j].item()
    #             if token in green_list:
    #                 count += 1
    #         counts[i] = count
    #         lengths[i] = n_tokens

    #     del clusters, inputs
    #     del sequences

    #     torch.cuda.empty_cache()

    #     z = (counts - self.gamma * lengths) / \
    #         torch.sqrt(self.gamma * lengths * (1 - self.gamma))
    #     return z

    def score_text_batch(self, batch_text):
        """
        Score a batch of text using the model and tokenizer.
        Args:
            batch_text: List of text samples to score.
            tokenizer: The tokenizer to use for scoring."""
        with torch.no_grad():
            inputs = self.tokenizer(batch_text, padding=True,
                                    return_tensors="pt").to(self.cluster_detector.device)
            hidden_states = self.cluster_detector(inputs)
            llr_scores = self.llr_raw(hidden_states, inputs)

        return llr_scores
