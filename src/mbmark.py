import torch
from functools import cached_property
import scipy.stats


class MbClusterDetector(torch.nn.Module):
    def __init__(self, model, unembedding_param_name):
        super(MbClusterDetector, self).__init__()
        self.model = model
        self.device = model.device
        self.model.eval()
        setattr(self.model, unembedding_param_name, torch.nn.Identity())

    def forward(self, x):
        outputs = self.model(**x)
        hidden_states = outputs.logits
        return hidden_states


class MbMark:
    def __init__(self, delta, gamma, seed, final_weight, model, tokenizer, unembedding_param_name,  mode="detect"):
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
                model, unembedding_param_name=unembedding_param_name)
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

    @torch.no_grad()
    def weighted_sum_raw(self, hidden_states, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Ignore BOS token
        # Labels are the next-token shifted version of input_ids
        labels = input_ids[:, 2:].clone().to(self.unembedding_matrix.device)
        attention_mask = attention_mask[:, 2:].to(
            self.unembedding_matrix.device)
        hidden_states = hidden_states[:, 1:-
                                      1].to(self.unembedding_matrix.device)

        delta_mat = torch.matmul(
            self.watermarking_matrix, self.final_weight)

        delta_mat = delta_mat.to(self.unembedding_matrix.device).to(
            self.unembedding_matrix.dtype)

        # Multiply by hidden states to get delta values
        deltas = torch.matmul(hidden_states, delta_mat.T)

        token_deltas = deltas.gather(
            dim=2, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Mask padding
        token_deltas = token_deltas * attention_mask  # (B, T)

        # Sum over tokens per sequence
        score_per_sequence = token_deltas.sum(dim=1)  # (B,)

        # Normalize
        lengths = attention_mask.sum(dim=1).float()  # (B,)

        score_per_sequence = score_per_sequence / torch.sqrt(lengths)

        return score_per_sequence

    @torch.no_grad()
    def llr_raw(self, hidden_states, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        labels = input_ids[:, 2:].clone().to(self.unembedding_matrix.device)
        attention_mask = attention_mask[:, 2:].to(
            self.unembedding_matrix.device)
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

        sum_ll_base = log_probs_base.sum(dim=1)
        sum_ll_marked = log_probs_marked.sum(dim=1)

        llr = sum_ll_marked - sum_ll_base

        llr = llr / attention_mask.sum(dim=1).float()

        # Clean up
        del inputs, hidden_states, log_probs_base, log_probs_marked
        del logits_base, logits_marked, labels
        del W_mod, delta_mat
        del attention_mask, mask
        del sum_ll_base, sum_ll_marked
        torch.cuda.empty_cache()
        return llr.cpu().float()
    

    @torch.no_grad()
    def llr_decomposed(self, hidden_states, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        labels = input_ids[:, 2:].clone().to(self.unembedding_matrix.device)
        attention_mask = attention_mask[:, 2:].to(self.unembedding_matrix.device)
        hidden_states = hidden_states[:, 1:-1].to(self.unembedding_matrix.device)

        delta_mat = torch.matmul(self.watermarking_matrix, self.final_weight)
        delta_mat = delta_mat.to(self.unembedding_matrix.device).to(self.unembedding_matrix.dtype)
        W_mod = self.unembedding_matrix + delta_mat

        # Token-specific unembedding vectors
        U_xt = self.unembedding_matrix[labels]  # (B, T, D)
        U_tilde_xt = W_mod[labels]              # (B, T, D)

        # First term: (U'_{x_t} - U_{x_t}) · h_t
        delta_term = ((U_tilde_xt - U_xt) * hidden_states).sum(dim=-1)  # (B, T)

        # Second term (flipped to make positive): log (Z_ref / Z_wm)
        logits_base = hidden_states @ self.unembedding_matrix.T  # (B, T, V)
        logits_marked = hidden_states @ W_mod.T                  # (B, T, V)

        logZ_base = torch.logsumexp(logits_base, dim=-1)         # (B, T)
        logZ_marked = torch.logsumexp(logits_marked, dim=-1)     # (B, T)
        partition_ratio_term = logZ_base - logZ_marked           # flipped (B, T)

        # Apply attention mask
        mask = attention_mask.bool()
        delta_term = delta_term.masked_fill(~mask, 0.0)
        partition_ratio_term = partition_ratio_term.masked_fill(~mask, 0.0)

        norm = attention_mask.sum(dim=1).float()  # (B,)

        delta_term = delta_term.sum(dim=1) / norm  # (B,)
        partition_ratio_term = partition_ratio_term.sum(dim=1) / norm

        return delta_term, partition_ratio_term
        

    def binomial_count(self, hidden_states, inputs):
        batch_size = inputs.input_ids.shape[0]
        selectors = torch.matmul(hidden_states.to(
            self.final_weight.dtype), self.final_weight.T)
        clusters = torch.argmax(selectors, dim=-1)

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
        return z

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
            # Compute the LLR score
            llr_scores = self.llr_raw(hidden_states, inputs)
            # llr_scores = (llr_scores - self.llr_mean) / self.llr_std

            # # Compute the binomial count score
            # binomial_scores = self.binomial_count(hidden_states, inputs)

            # # Compute the weighted sum score
            # weighted_sum_scores = self.weighted_sum_raw(
            #     hidden_states, inputs)
            # weighted_sum_scores = (weighted_sum_scores -
            #                        self.weighted_sum_mean) / self.weighted_sum_std

        return llr_scores


class MbMark2:

    model_to_hs_norm = {
        "meta-llama/Llama-2-7b-hf": 118.0,
        "mistralai/Mistral-7B-v0.3": 365.22
    }

    def __init__(self, seed, model, tokenizer, unembedding_param_name,  mode="detect"):
        self.seed = seed
        self.vocab_size = len(tokenizer)
        self.tokenizer = tokenizer
        self.unembedding_param_name = unembedding_param_name
        self.unembedding_matrix = getattr(
            model, unembedding_param_name).weight.data
        self.hs_norm = self.model_to_hs_norm[model.config.name_or_path]
        self.unembedding_matrix.requires_grad = False
        assert self.unembedding_matrix.shape[0] == self.vocab_size, "Unembedding matrix and watermarking matrix must have the same vocab size"
        if mode == "detect":
            self.cluster_detector = MbClusterDetector(
                model, unembedding_param_name=unembedding_param_name)
            self.cluster_detector.eval()
        elif mode == "generate":
            self.watermark_model(model)

    @cached_property
    def delta_matrix(self):
        # Generate random matrix of size (V, hidden_size) wiht seed
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        hidden_size = self.unembedding_matrix.shape[1]
        delta_mat = torch.randn(
            self.vocab_size, hidden_size, generator=rng)

        # Divide by sqrt(hidden_size) to normalize
        delta_mat = delta_mat / (self.hs_norm * 1.8)

        return delta_mat

    def watermark_model(self, model):
        delta_mat = self.delta_matrix.to(
            self.unembedding_matrix.device).to(self.unembedding_matrix.dtype)

        with torch.no_grad():
            augmented_unembedding = self.unembedding_matrix.clone() + delta_mat
            getattr(model, self.unembedding_param_name).weight.data.copy_(
                augmented_unembedding)

        self.model = model

    @torch.no_grad()
    def weighted_sum_raw(self, hidden_states, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Ignore BOS token
        # Labels are the next-token shifted version of input_ids
        labels = input_ids[:, 2:].clone().to(self.unembedding_matrix.device)
        attention_mask = attention_mask[:, 2:].to(
            self.unembedding_matrix.device)
        hidden_states = hidden_states[:, 1:-
                                      1].to(self.unembedding_matrix.device)

        delta_mat = self.delta_matrix.to(self.unembedding_matrix.device).to(
            self.unembedding_matrix.dtype)

        # Multiply by hidden states to get delta values
        deltas = torch.matmul(hidden_states, delta_mat.T)

        token_deltas = deltas.gather(
            dim=2, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Mask padding
        token_deltas = token_deltas * attention_mask  # (B, T)

        # Sum over tokens per sequence
        score_per_sequence = token_deltas.sum(dim=1)  # (B,)

        # Normalize
        lengths = attention_mask.sum(dim=1).float()  # (B,)

        score_per_sequence = score_per_sequence / torch.sqrt(lengths)

        return score_per_sequence

    @torch.no_grad()
    def llr_raw(self, hidden_states, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        labels = input_ids[:, 2:].clone().to(self.unembedding_matrix.device)
        attention_mask = attention_mask[:, 2:].to(
            self.unembedding_matrix.device)
        hidden_states = hidden_states[:, 1:-
                                      1].to(self.unembedding_matrix.device)

        delta_mat = self.delta_matrix.to(self.unembedding_matrix.device).to(
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

        sum_ll_base = log_probs_base.sum(dim=1)
        sum_ll_marked = log_probs_marked.sum(dim=1)

        llr = sum_ll_marked - sum_ll_base

        llr = llr / attention_mask.sum(dim=1).float()

        # Clean up
        del inputs, hidden_states, log_probs_base, log_probs_marked
        del logits_base, logits_marked, labels
        del W_mod, delta_mat
        del attention_mask, mask
        del sum_ll_base, sum_ll_marked
        torch.cuda.empty_cache()
        return llr.cpu().float()

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
            # Compute the LLR score
            llr_scores = self.llr_raw(hidden_states, inputs)

        return llr_scores


    @torch.no_grad()
    def llr_decomposed(self, hidden_states, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        labels = input_ids[:, 2:].clone().to(self.unembedding_matrix.device)
        attention_mask = attention_mask[:, 2:].to(self.unembedding_matrix.device)
        hidden_states = hidden_states[:, 1:-1].to(self.unembedding_matrix.device)

        delta_mat = self.delta_matrix.to(self.unembedding_matrix.device).to(self.unembedding_matrix.dtype)
        W_mod = self.unembedding_matrix + delta_mat

        # Token-specific unembedding vectors
        U_xt = self.unembedding_matrix[labels]  # (B, T, D)
        U_tilde_xt = W_mod[labels]              # (B, T, D)

        # First term: (U'_{x_t} - U_{x_t}) · h_t
        delta_term = ((U_tilde_xt - U_xt) * hidden_states).sum(dim=-1)  # (B, T)

        # Second term (flipped to make positive): log (Z_ref / Z_wm)
        logits_base = hidden_states @ self.unembedding_matrix.T  # (B, T, V)
        logits_marked = hidden_states @ W_mod.T                  # (B, T, V)

        logZ_base = torch.logsumexp(logits_base, dim=-1)         # (B, T)
        logZ_marked = torch.logsumexp(logits_marked, dim=-1)     # (B, T)
        partition_ratio_term = logZ_base - logZ_marked           # flipped (B, T)

        # Apply attention mask
        mask = attention_mask.bool()
        delta_term = delta_term.masked_fill(~mask, 0.0)
        partition_ratio_term = partition_ratio_term.masked_fill(~mask, 0.0)

        norm = attention_mask.sum(dim=1).float()  # (B,)

        delta_term = delta_term.sum(dim=1) / norm  # (B,)
        partition_ratio_term = partition_ratio_term.sum(dim=1) / norm

        return delta_term, partition_ratio_term