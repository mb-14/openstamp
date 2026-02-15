"""
PyTorch implementation of MiniBatchKMeans.

Accepts torch.Tensor only. Default device is CPU.
API: fit, predict, transform, fit_predict, cluster_centers_, labels_, inertia_

GPU support: When device='cuda', the full dataset stays on CPU;
only each minibatch is moved to GPU during training. This allows datasets
larger than GPU memory. Init and compute_labels use chunked processing.
"""

import warnings
import torch
import torch.nn.functional as F
from tqdm import trange


def _ensure_tensor(X, name="X"):
    """Raise TypeError if X is not a torch.Tensor."""
    if not isinstance(X, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(X).__name__}."
        )
    return X


# ---------------------------------------------------------------------------
# Phase 1: Core Tensor Operations
# ---------------------------------------------------------------------------


def pairwise_squared_distances(X, C):
    """Batched squared L2 distances: ||X - C||^2.

    X : (n_samples, n_features)
    C : (n_clusters, n_features)
    Returns : (n_samples, n_clusters)
    """
    x_sq = (X ** 2).sum(dim=1, keepdim=True)
    c_sq = (C ** 2).sum(dim=1)  # (n_clusters,)
    dist_sq = x_sq - 2 * X @ C.T + c_sq
    return dist_sq


def labels_inertia(X, centers, sample_weight=None, return_inertia=True):
    """E-step: assign labels and optionally compute inertia.

    Returns
    -------
    labels : (n_samples,) int64
    inertia : float, only if return_inertia=True
    """
    dist_sq = pairwise_squared_distances(X, centers)
    labels = dist_sq.argmin(dim=1)

    if not return_inertia:
        return labels

    if sample_weight is None:
        sample_weight = torch.ones(X.shape[0], dtype=X.dtype, device=X.device)

    # Inertia = sum over samples of (sample_weight[i] * squared_dist to assigned center)
    centers_gathered = centers[labels]  # (n_samples, n_features)
    sq_dists = ((X - centers_gathered) ** 2).sum(dim=1)
    inertia = (sq_dists * sample_weight).sum().item()
    return labels, inertia


def lloyd_iter_dense(X, centers_old, sample_weight=None):
    """Single Lloyd iteration: E-step + M-step with empty cluster handling.

    Returns
    -------
    labels : (n_samples,) int64
    centers_new : (n_clusters, n_features)
    center_shift : (n_clusters,)
    """
    n_samples, n_features = X.shape
    n_clusters = centers_old.shape[0]

    if sample_weight is None:
        sample_weight = torch.ones(n_samples, dtype=X.dtype, device=X.device)

    dist_sq = pairwise_squared_distances(X, centers_old)
    labels = dist_sq.argmin(dim=1)

    # M-step: weighted mean per cluster
    one_hot = F.one_hot(labels, n_clusters).to(X.dtype)
    weighted = one_hot.T @ (X * sample_weight.unsqueeze(1))
    weights = one_hot.T @ sample_weight

    centers_new = centers_old.clone()
    centers_new.copy_(weighted / weights.clamp(min=1e-10).unsqueeze(1))

    # Handle empty clusters: relocate to farthest points
    relocate_empty_clusters(
        X, centers_old, centers_new, weights, labels, sample_weight
    )

    # Re-average after relocation (weights may have changed)
    one_hot = F.one_hot(labels, n_clusters).to(X.dtype)
    weighted = one_hot.T @ (X * sample_weight.unsqueeze(1))
    weights = one_hot.T @ sample_weight
    centers_new.copy_(weighted / weights.clamp(min=1e-10).unsqueeze(1))

    # Center shift
    center_shift = (centers_new - centers_old).norm(dim=1)

    return labels, centers_new, center_shift


def minibatch_update(X, centers_old, centers_new, weight_sums, labels, sample_weight=None):
    """Incremental center update for MiniBatchKMeans.

    Modifies centers_new and weight_sums in-place.
    """
    n_samples, n_features = X.shape
    n_clusters = centers_old.shape[0]

    if sample_weight is None:
        sample_weight = torch.ones(n_samples, dtype=X.dtype, device=X.device)

    # For each cluster: center_new = (center_old * weight_sum + batch_sum) / (weight_sum + batch_weight)
    # Avoid large (n_samples, n_clusters) one-hot by using index_add_/bincount
    batch_weighted = torch.zeros(
        (n_clusters, n_features), device=X.device, dtype=X.dtype
    )
    batch_weighted.index_add_(0, labels, X * sample_weight.unsqueeze(1))
    batch_weights = torch.bincount(
        labels, weights=sample_weight, minlength=n_clusters
    ).to(X.dtype)

    for k in range(n_clusters):
        wsum = batch_weights[k].item()
        if wsum > 0:
            # Undo previous scaling, add new points, rescale
            centers_new[k] = centers_old[k] * weight_sums[k] + batch_weighted[k]
            weight_sums[k] = weight_sums[k] + wsum
            centers_new[k] = centers_new[k] / weight_sums[k]
        else:
            centers_new[k] = centers_old[k]


def relocate_empty_clusters(
    X, centers_old, centers_new, weight_in_clusters, labels, sample_weight
):
    """Relocate empty clusters by reassigning farthest points. Modifies labels in-place.

    Caller should re-compute centers from the updated labels.
    """
    empty_mask = weight_in_clusters == 0
    n_empty = empty_mask.sum().item()

    if n_empty == 0:
        return

    centers_gathered = centers_old[labels]
    dist_sq = ((X - centers_gathered) ** 2).sum(dim=1) * sample_weight

    if dist_sq.max().item() == 0:
        return

    _, far_indices = torch.topk(dist_sq, min(n_empty, dist_sq.numel()))
    empty_indices = torch.where(empty_mask)[0]
    n_reassign = min(n_empty, len(far_indices))

    for i in range(n_reassign):
        far_idx = far_indices[i].item()
        new_cluster_id = empty_indices[i].item()
        labels[far_idx] = new_cluster_id


# ---------------------------------------------------------------------------
# Phase 2: K-means++ Initialization
# ---------------------------------------------------------------------------


def _kmeans_plusplus(
    X,
    n_clusters,
    x_squared_norms,
    generator,
    projection_dim=None,
    subsample=None,
    verbose=False,
):
    """K-means++ initialization. Returns centers (n_clusters, n_features)."""
    n_samples, n_features = X.shape

    if subsample is not None and subsample < n_samples:
        subsample = max(int(subsample), n_clusters)
        perm = torch.randperm(n_samples, device=X.device, generator=generator)[:subsample]
        X_work = X[perm]
        x_squared_norms_work = x_squared_norms[perm]
        index_map = perm
    else:
        X_work = X
        x_squared_norms_work = x_squared_norms
        index_map = torch.arange(n_samples, device=X.device)

    n_samples_work = X_work.shape[0]

    if projection_dim is not None and projection_dim < X_work.shape[1]:
        # Simple random projection (Gaussian)
        proj = torch.randn(
            n_features, projection_dim, device=X.device, dtype=X.dtype, generator=generator
        ) / (projection_dim ** 0.5)
        X_dist = X_work @ proj
        x_squared_norms_dist = (X_dist ** 2).sum(dim=1)
    else:
        X_dist = X_work
        x_squared_norms_dist = x_squared_norms_work

    centers = torch.empty((n_clusters, n_features), dtype=X.dtype, device=X.device)
    n_local_trials = 2 + int(torch.log(torch.tensor(n_clusters, dtype=torch.float32)).item())
    # First center: random
    center_id = torch.randint(
        n_samples_work, (1,), device=X.device, generator=generator
    ).item()
    centers[0] = X_work[center_id]

    first_center = X_dist[center_id : center_id + 1]
    closest_dist_sq = pairwise_squared_distances(first_center, X_dist).squeeze(0)
    current_pot = closest_dist_sq.sum().item()

    # Remaining centers
    iterator = trange(1, n_clusters, desc="k-means++") if verbose else range(1, n_clusters)
    for c in iterator:
        rand_vals = torch.rand(n_local_trials, device=X.device, generator=generator) * current_pot
        cumsum = torch.cumsum(closest_dist_sq, dim=0)
        candidate_ids = torch.searchsorted(cumsum, rand_vals, right=True)
        candidate_ids = torch.clamp(candidate_ids, 0, closest_dist_sq.numel() - 1)

        dist_to_candidates = pairwise_squared_distances(
            X_dist[candidate_ids], X_dist
        )
        candidates_dist = torch.minimum(
            closest_dist_sq.unsqueeze(0).expand_as(dist_to_candidates),
            dist_to_candidates,
        )
        candidates_pot = candidates_dist.sum(dim=1)

        best_idx = candidates_pot.argmin().item()
        current_pot = candidates_pot[best_idx].item()
        closest_dist_sq = candidates_dist[best_idx]
        best_candidate = candidate_ids[best_idx].item()

        centers[c] = X_work[best_candidate]

    return centers


# ---------------------------------------------------------------------------
# Phase 3 & 4: Labels/Inertia and Mini-batch Step
# ---------------------------------------------------------------------------


def _tolerance(X, tol):
    """Tolerance dependent on dataset variance."""
    if tol == 0:
        return 0.0
    variances = X.var(dim=0)
    return variances.mean().item() * tol


def _labels_inertia(X, centers, sample_weight=None, return_inertia=True):
    """Labels and inertia (PyTorch version)."""
    return labels_inertia(X, centers, sample_weight, return_inertia)


def _mini_batch_step(
    X,
    centers,
    centers_new,
    weight_sums,
    generator,
    random_reassign=False,
    reassignment_ratio=0.01,
    verbose=False,
):
    """One minibatch step. Returns inertia (float)."""
    labels, inertia = _labels_inertia(X, centers, return_inertia=True)
    sample_weight = torch.ones(X.shape[0], dtype=X.dtype, device=X.device)

    minibatch_update(X, centers, centers_new, weight_sums, labels, sample_weight)

    if random_reassign and reassignment_ratio > 0:
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()
        n_to_reassign = to_reassign.sum().item()

        if n_to_reassign > 0.5 * X.shape[0]:
            # Limit reassignments: keep clusters with highest weight among those marked
            masked_indices = torch.where(to_reassign)[0]
            masked_weights = weight_sums[to_reassign]
            n_keep = max(0, n_to_reassign - int(0.5 * X.shape[0]))
            _, top_in_masked = torch.topk(masked_weights, min(n_keep, len(masked_indices)))
            indices_dont_reassign = masked_indices[top_in_masked]
            to_reassign[indices_dont_reassign] = False
            n_to_reassign = to_reassign.sum().item()

        if n_to_reassign > 0:
            perm = torch.randperm(X.shape[0], device=X.device, generator=generator)
            new_center_indices = perm[:n_to_reassign]
            if verbose:
                print(f"[MiniBatchKMeans] Reassigning {n_to_reassign} cluster centers.")
            centers_new[to_reassign] = X[new_center_indices]

            non_empty = weight_sums[~to_reassign]
            if non_empty.numel() > 0:
                weight_sums[to_reassign] = non_empty.min()
            else:
                weight_sums[to_reassign] = 1.0

    return inertia


# ---------------------------------------------------------------------------
# Phase 5: MiniBatchKMeans Class
# ---------------------------------------------------------------------------


class MiniBatchKMeans:
    """
    Mini-Batch K-Means clustering (PyTorch).

    Accepts torch.Tensor only. Default device is CPU.
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        max_iter=100,
        batch_size=1024,
        verbose=0,
        compute_labels=True,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init="auto",
        reassignment_ratio=0.01,
        init_projection_dim=None,
        init_subsample=None,
        device="cpu",
        dtype=torch.float32,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = bool(verbose)
        self.compute_labels = compute_labels
        self.random_state = random_state
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.init_size = init_size
        self.n_init = n_init
        self.reassignment_ratio = reassignment_ratio
        self.init_projection_dim = init_projection_dim
        self.init_subsample = init_subsample
        self.device = device
        self.dtype = dtype

    def _get_generator(self, device=None):
        """Generator for device-specific tensors (e.g. batch on GPU)."""
        device = device or self.device
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device=device)
        g.manual_seed(int(self.random_state))
        return g

    def _get_cpu_generator(self):
        """Generator for index tensors (always CPU when X is on CPU)."""
        if self.random_state is None:
            return None
        if isinstance(self.random_state, torch.Generator):
            return self.random_state
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.random_state))
        return g

    def _is_cpu_device(self):
        """True if device is CPU."""
        dev = self.device
        if isinstance(dev, str):
            return dev == "cpu"
        return dev.type == "cpu"

    def _check_params_vs_input(self, X, default_n_init=3):
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )
        self._tol = _tolerance(X, self.tol)

        if self.init not in ("k-means++", "random"):
            raise ValueError(
                f"init must be 'k-means++' or 'random', got {self.init!r}."
            )

        if self.n_init == "auto":
            self._n_init = 1 if self.init == "k-means++" else default_n_init
        else:
            self._n_init = self.n_init

        self._batch_size = min(self.batch_size, X.shape[0])
        self._init_size = self.init_size
        if self._init_size is None:
            self._init_size = 3 * self._batch_size
            if self._init_size < self.n_clusters:
                self._init_size = 3 * self.n_clusters
        elif self._init_size < self.n_clusters:
            warnings.warn(
                f"init_size={self._init_size} should be larger than "
                f"n_clusters={self.n_clusters}. Setting to 3*n_clusters.",
                RuntimeWarning,
            )
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])

        if self.reassignment_ratio < 0:
            raise ValueError("reassignment_ratio should be >= 0.")
        if self.init_projection_dim is not None and self.init_projection_dim <= 0:
            raise ValueError("init_projection_dim should be > 0.")
        if self.init_subsample is not None and self.init_subsample < self.n_clusters:
            raise ValueError("init_subsample should be >= n_clusters.")

    def _init_centroids(self, X, x_squared_norms, init, generator, init_size=None):
        n_samples = X.shape[0]
        if init_size is not None and init_size < n_samples:
            perm = torch.randperm(n_samples, device=X.device, generator=generator)
            idx = perm[:init_size]
            X = X[idx]
            x_squared_norms = x_squared_norms[idx]
            n_samples = X.shape[0]

        if init == "k-means++":
            centers = _kmeans_plusplus(
                X,
                self.n_clusters,
                x_squared_norms,
                generator,
                projection_dim=self.init_projection_dim,
                subsample=self.init_subsample,
                verbose=self.verbose,
            )
        elif init == "random":
            perm = torch.randperm(n_samples, device=X.device, generator=generator)
            seeds = perm[: self.n_clusters]
            centers = X[seeds].clone()
        else:
            raise ValueError(f"init must be 'k-means++' or 'random', got {init!r}.")
        return centers

    def _mini_batch_convergence(
        self, step, n_steps, n_samples, centers_squared_diff, batch_inertia
    ):
        batch_inertia = batch_inertia / self._batch_size
        step = step + 1

        if step == 1:
            if self.verbose:
                print(f"Minibatch step {step}/{n_steps}: mean batch inertia: {batch_inertia}")
            return False

        if self._ewa_inertia is None:
            self._ewa_inertia = batch_inertia
        else:
            alpha = self._batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1.0)
            self._ewa_inertia = self._ewa_inertia * (1 - alpha) + batch_inertia * alpha

        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch inertia: "
                f"{batch_inertia}, ewa inertia: {self._ewa_inertia}"
            )

        if self._tol > 0.0 and centers_squared_diff <= self._tol:
            if self.verbose:
                print(f"Converged (small centers change) at step {step}/{n_steps}")
            return True

        if self._ewa_inertia_min is None or self._ewa_inertia < self._ewa_inertia_min:
            self._no_improvement = 0
            self._ewa_inertia_min = self._ewa_inertia
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    f"Converged (lack of improvement in inertia) at step {step}/{n_steps}"
                )
            return True

        return False

    def _random_reassign(self):
        self._n_since_last_reassign += self._batch_size
        if (self._counts == 0).any().item() or self._n_since_last_reassign >= (
            10 * self.n_clusters
        ):
            self._n_since_last_reassign = 0
            return True
        return False

    def fit(self, X):
        """Compute the centroids on X by chunking into mini-batches.

        When device='cuda', X stays on CPU; only each batch is moved to GPU.
        This allows datasets larger than GPU memory.
        """
        X = _ensure_tensor(X)
        X = X.to(dtype=self.dtype)
        if X.dim() != 2:
            raise ValueError("X must be 2D.")

        self._check_params_vs_input(X)
        device = self.device
        cpu_gen = self._get_cpu_generator()
        dev_gen = self._get_generator()
        n_samples, n_features = X.shape

        x_squared_norms = (X ** 2).sum(dim=1)

        # Validation set for init (indices on CPU)
        if cpu_gen is not None:
            valid_idx = torch.randint(
                0, n_samples, (self._init_size,), device="cpu", generator=cpu_gen
            )
        else:
            valid_idx = torch.randint(0, n_samples, (self._init_size), device="cpu")
        X_valid = X[valid_idx].to(device)

        # Init subset: move to device so full X stays on CPU
        if cpu_gen is not None:
            init_perm = torch.randperm(n_samples, device="cpu", generator=cpu_gen)[
                : self._init_size
            ]
        else:
            init_perm = torch.randperm(n_samples, device="cpu")[: self._init_size]
        X_init = X[init_perm].to(device)
        x_sq_init = x_squared_norms[init_perm].to(device)

        best_inertia = None
        for init_idx in range(self._n_init):
            if self.verbose:
                print(f"Init {init_idx + 1}/{self._n_init} with method {self.init}")

            cluster_centers = self._init_centroids(
                X_init,
                x_squared_norms=x_sq_init,
                init=self.init,
                generator=dev_gen,
                init_size=None,
            )

            _, inertia = _labels_inertia(X_valid, cluster_centers, return_inertia=True)
            inertia_val = inertia

            if self.verbose:
                print(f"Inertia for init {init_idx + 1}/{self._n_init}: {inertia_val}")
            if best_inertia is None or inertia_val < best_inertia:
                init_centers = cluster_centers
                best_inertia = inertia_val

        centers = init_centers
        centers_new = torch.empty_like(centers)
        self._counts = torch.zeros(self.n_clusters, dtype=X.dtype, device=device)
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0
        self._n_since_last_reassign = 0

        n_steps = (self.max_iter * n_samples) // self._batch_size

        for i in range(n_steps):
            if cpu_gen is not None:
                mb_idx = torch.randint(
                    0, n_samples, (self._batch_size,), device="cpu", generator=cpu_gen
                )
            else:
                mb_idx = torch.randint(0, n_samples, (self._batch_size,), device="cpu")
            X_batch = X[mb_idx].to(device)

            batch_inertia = _mini_batch_step(
                X_batch,
                centers,
                centers_new,
                self._counts,
                generator=dev_gen,
                random_reassign=self._random_reassign(),
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose,
            )

            if self._tol > 0.0:
                centers_squared_diff = ((centers_new - centers) ** 2).sum().item()
            else:
                centers_squared_diff = 0.0

            centers, centers_new = centers_new, centers

            if self._mini_batch_convergence(
                i, n_steps, n_samples, centers_squared_diff, batch_inertia
            ):
                break

        self.cluster_centers_ = centers
        self.n_steps_ = i + 1
        self.n_iter_ = int(((i + 1) * self._batch_size + n_samples - 1) // n_samples)

        if self.compute_labels:
            self.labels_, self.inertia_ = self._labels_inertia_chunked(X, self.cluster_centers_, device)
                
        else:
            self.inertia_ = self._ewa_inertia * n_samples if self._ewa_inertia is not None else 0.0

        return self

    @torch.no_grad()
    def _labels_inertia_chunked(self, X, centers, device):
        """Compute labels and inertia in chunks (avoids loading full X on GPU)."""
        n_samples = X.shape[0]
        chunk_size = self._batch_size
        labels_list = []
        inertia_total = 0.0

        for start in trange(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            X_chunk = X[start:end].to(device)
            chunk_labels, chunk_inertia = _labels_inertia(
                X_chunk, centers, return_inertia=True
            )
            labels_list.append(chunk_labels.cpu())
            inertia_total += chunk_inertia

        return torch.cat(labels_list), inertia_total

    def predict(self, X):
        """Predict the closest cluster each sample belongs to.

        Uses chunked processing when device is not CPU (avoids OOM for large X).
        """
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        X = _ensure_tensor(X)
        X = X.to(dtype=self.dtype)
        if X.dim() != 2:
            raise ValueError("X must be 2D.")
        if self._is_cpu_device():
            return _labels_inertia(
                X, self.cluster_centers_, return_inertia=False
            )
        return self._predict_chunked(X)

    def _predict_chunked(self, X):
        """Chunked predict for GPU (keeps X on CPU)."""
        chunk_size = getattr(self, "_batch_size", 1024)
        chunk_size = min(chunk_size, X.shape[0])
        labels_list = []
        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])
            X_chunk = X[start:end].to(self.device)
            chunk_labels = _labels_inertia(
                X_chunk, self.cluster_centers_, return_inertia=False
            )
            labels_list.append(chunk_labels.cpu())
        return torch.cat(labels_list)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        Uses chunked processing when device is not CPU (avoids OOM for large X).
        """
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        X = _ensure_tensor(X)
        X = X.to(dtype=self.dtype)
        if X.dim() != 2:
            raise ValueError("X must be 2D.")
        if self._is_cpu_device():
            return torch.sqrt(
                pairwise_squared_distances(X, self.cluster_centers_).clamp(min=0)
            )
        return self._transform_chunked(X)

    def _transform_chunked(self, X):
        """Chunked transform for GPU (keeps X on CPU)."""
        chunk_size = getattr(self, "_batch_size", 1024)
        chunk_size = min(chunk_size, X.shape[0])
        dists_list = []
        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])
            X_chunk = X[start:end].to(self.device)
            d = torch.sqrt(
                pairwise_squared_distances(X_chunk, self.cluster_centers_).clamp(min=0)
            )
            dists_list.append(d.cpu())
        return torch.cat(dists_list)

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample."""
        return self.fit(X).labels_

    def score(self, X):
        """Opposite of the value of inertia.

        Uses chunked processing when device is not CPU (avoids OOM for large X).
        """
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        X = _ensure_tensor(X)
        X = X.to(dtype=self.dtype)
        if X.dim() != 2:
            raise ValueError("X must be 2D.")
        if self._is_cpu_device():
            _, inertia = _labels_inertia(
                X, self.cluster_centers_, return_inertia=True
            )
            return -inertia
        _, inertia = self._labels_inertia_chunked(X, self.cluster_centers_, self.device)
        return -inertia
