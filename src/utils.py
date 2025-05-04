import torch
from functools import cached_property
import scipy.stats


class MbClusterDetector(torch.nn.Module):
    def __init__(self, model, unembedding_param_name, final_weight):
        super(MbClusterDetector, self).__init__()
        self.model = model
        self.device = model.device
        self.model.eval()
        setattr(self.model, unembedding_param_name, torch.nn.Identity())
        self.final_weight = final_weight
        self.final_weight.requires_grad = False
        self.final_weight = self.final_weight.to(
            self.model.device)

    def forward(self, x, get_hidden_states=False):
        outputs = self.model(**x)
        hidden_states = outputs.logits
        if get_hidden_states:
            return hidden_states

        hidden_states = hidden_states.to(self.final_weight.dtype)
        selectors = torch.matmul(
            hidden_states, self.final_weight.T)
        clusters = torch.argmax(selectors, dim=-1)
        return clusters


class MbMark:
    def __init__(self, delta, gamma, seed, final_weight, model, tokenizer, unembedding_param_name, mode="detect"):
        self.seed = seed
        self.delta = delta
        self.gamma = gamma
        self.final_weight = final_weight
        self.n_clusters = final_weight.size(0)
        self.vocab_size = len(tokenizer)
        self.tokenizer = tokenizer
        self.unembedding_param_name = unembedding_param_name
        self.unembedding_matrix = getattr(
            model, unembedding_param_name).weight.data
        self.unembedding_matrix.requires_grad = False
        assert self.unembedding_matrix.shape[0] == self.vocab_size, "Unembedding matrix and watermarking matrix must have the same vocab size"
        if mode == "detect":
            self.cluster_detector = MbClusterDetector(
                model, unembedding_param_name=unembedding_param_name, final_weight=final_weight)
            self.cluster_detector.eval()
        elif mode == "generate":
            self.watermark_model(model)

    def prf_lookup(self, cluster):
        return self.seed * cluster

    def get_partition(self, cluster):
        rng = torch.Generator()
        prf_key = self.prf_lookup(cluster)
        rng.manual_seed(prf_key % (2**64 - 1))

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, generator=rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        redlist_ids = vocab_permutation[greenlist_size:]
        return greenlist_ids, redlist_ids

    @cached_property
    def watermarking_matrix(self):
        watermark_matrix = torch.zeros(
            self.vocab_size, self.n_clusters)
        for i in range(self.n_clusters):
            greenlist_ids, redlist_ids = self.get_partition(i)
            watermark_matrix[greenlist_ids, i] = self.delta
            watermark_matrix[redlist_ids, i] = 0.0

        return watermark_matrix

    def watermark_model(self, model):
        # Type cast during integeration
        final_weight = self.final_weight.to(
            self.unembedding_matrix.device).to(self.unembedding_matrix.dtype)
        watermarking_matrix = self.watermarking_matrix.to(
            self.unembedding_matrix.device).to(self.unembedding_matrix.dtype)

        delta_mat = torch.matmul(watermarking_matrix, final_weight)

        with torch.no_grad():
            augmented_unembedding = self.unembedding_matrix.clone() + delta_mat
            getattr(model, self.unembedding_param_name).weight.data.copy_(
                augmented_unembedding)

        self.model = model

    def score_lrt(self, batch_text, mean, std):
        lrts = self.lrt(batch_text)
        scores = (lrts - mean) / std
        p_values = 1 - torch.distributions.Normal(0, 1).cdf(scores)
        return scores, p_values

    def lrt(self, batch_text):
        with torch.no_grad():
            inputs = self.tokenizer(batch_text, padding=True,
                                    return_tensors="pt").to(self.cluster_detector.device)

            hidden_states = self.cluster_detector(
                inputs, get_hidden_states=True)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # Ignore BOS token
            # Labels are the next-token shifted version of input_ids
            labels = input_ids[:, 2:].clone().to(self.unembedding_matrix.device)
            attention_mask = attention_mask[:, 2:].to(self.unembedding_matrix.device)
            hidden_states = hidden_states[:, 1:-
                                          1].to(self.unembedding_matrix.device)

            delta_mat = torch.matmul(
                self.watermarking_matrix, self.final_weight)

            delta_mat = delta_mat.to(self.unembedding_matrix.device).to(
                self.unembedding_matrix.dtype)

            W_mod = self.unembedding_matrix + delta_mat
            logits_base = hidden_states @ self.unembedding_matrix.T
            logits_marked = hidden_states @ W_mod.T        # (B, T, V)

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

            # Average over valid tokens
            avg_ll_base = log_probs_base.sum(dim=1) / mask.sum(dim=1)
            avg_ll_marked = log_probs_marked.sum(dim=1) / mask.sum(dim=1)
            
            log_ppl_ratio = avg_ll_marked - avg_ll_base
            # Clean up
            del inputs, hidden_states, log_probs_base, log_probs_marked
            del logits_base, logits_marked, labels
            del W_mod, delta_mat
            del attention_mask, mask
            del avg_ll_base, avg_ll_marked
            torch.cuda.empty_cache()
            return log_ppl_ratio.cpu().float()
        
    def boosting_score(self, batch_text):
        with torch.no_grad():
            inputs = self.tokenizer(batch_text, padding=True,
                                    return_tensors="pt").to(self.cluster_detector.device)

            hidden_states = self.cluster_detector(
                inputs, get_hidden_states=True)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # Ignore BOS token
            labels = input_ids[:, 2:].clone().to(self.unembedding_matrix.device)
            attention_mask = attention_mask[:, 2:].to(self.unembedding_matrix.device)
            hidden_states = hidden_states[:, 1:-1].to(self.unembedding_matrix.device)

            # delta_mat: shape (V, d)
            delta_mat = torch.matmul(self.watermarking_matrix, self.final_weight)
            delta_mat = delta_mat.to(self.unembedding_matrix.device).to(self.unembedding_matrix.dtype)

            # Compute logit boost directly: (B, T, V)
            delta_logits = hidden_states @ delta_mat.T

            # Get the delta logit for the actual token: (B, T)
            logit_boost = delta_logits.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            # Mask out padding tokens
            mask = attention_mask.bool()
            logit_boost = logit_boost.masked_fill(~mask, 0.0)

            # Sum across valid tokens
            total_logit_boost = logit_boost.sum(dim=1)

            # Clean up
            del inputs, hidden_states, delta_logits, labels, attention_mask, mask
            del delta_mat
            torch.cuda.empty_cache()

            return total_logit_boost.cpu().float()

    def score_text_batch(self, batch_text):
        """
        Score a batch of text using the model and tokenizer.
        Args:
            batch_text: List of text samples to score.
            tokenizer: The tokenizer to use for scoring."""
        batch_size = len(batch_text)
        with torch.no_grad():
            inputs = self.tokenizer(batch_text, padding=True,
                                    return_tensors="pt").to(self.cluster_detector.device)
            clusters = self.cluster_detector(inputs)
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
                    green_list = torch.arange(self.vocab_size)[
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
        p_values = torch.tensor(scipy.stats.binom.sf(
            counts.numpy(), lengths.numpy(), self.gamma))
        return z, p_values


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
                batch_text, padding=True, return_tensors="pt").to(self.model.device)
            input_ids = inputs.input_ids

            # Forward pass
            logits = self.model(**inputs).logits  # (B, T, V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Next-token predictions
            shifted_input_ids = input_ids[:, 1:]              # (B, T-1)
            shifted_log_probs = log_probs[:, :-1, :]          # (B, T-1, V)
            log_probs_seq = torch.gather(
                shifted_log_probs, 2, shifted_input_ids.unsqueeze(-1)
            ).squeeze(-1)                                     # (B, T-1)

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
                    retain_graph=True,  # <-- crucial for reusing the graph
                    create_graph=False,
                    only_inputs=True,
                    allow_unused=True
                )[0]

                if grad_i is None:
                    raise RuntimeError(f"Gradient is None for sample {i}.")
                # Detach to avoid holding graph refs
                grads.append(grad_i.view(-1).detach())

            grads = torch.stack(grads, dim=0)  # (B, D)
            keys = self.watermark_key(weight.shape).view(
                1, -1).to(grads.device)  # (1, D)

            # Compute test statistics
            dots = (keys * grads).sum(dim=1)              # (B,)
            norms = grads.norm(dim=1)                     # (B,)
            psi = dots / (self.sigma * norms)             # (B,)
            z_scores = psi
            p_values = 1 - torch.distributions.Normal(0, 1).cdf(psi)

            self.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            return z_scores, p_values
