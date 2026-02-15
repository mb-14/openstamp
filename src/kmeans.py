import warnings
import time
import numpy as np
from sklearn.metrics.pairwise import _euclidean_distances, euclidean_distances
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import check_array, check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.utils.parallel import (
    _get_threadpool_controller,
    _threadpool_controller_decorator,
)

from sklearn.cluster._k_means_common import _inertia_dense
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
from sklearn.cluster._k_means_minibatch import _minibatch_update_dense
from tqdm import trange


def _kmeans_plusplus(
    X,
    n_clusters,
    x_squared_norms,
    random_state,
    projection_dim=None,
    subsample=None,
):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    # Subsample 200k points to speed up the calculation of the initial centers
    # idx = random_state.choice(X.shape[0], min(200000, X.shape[0]), replace=False)
    # X = X[idx]
    # x_squared_norms = x_squared_norms[idx]

    n_samples, n_features = X.shape

    if subsample is not None and subsample < n_samples:
        subsample = max(int(subsample), n_clusters)
        subsample_indices = random_state.choice(n_samples, subsample, replace=False)
        X_work = X[subsample_indices]
        x_squared_norms_work = x_squared_norms[subsample_indices]
        index_map = subsample_indices
    else:
        X_work = X
        x_squared_norms_work = x_squared_norms
        index_map = np.arange(n_samples)

    n_samples = X_work.shape[0]

    if projection_dim is not None and projection_dim < X_work.shape[1]:
        projector = SparseRandomProjection(
            n_components=projection_dim,
            random_state=random_state,
        )
        X_dist = projector.fit_transform(X_work)
        x_squared_norms_dist = row_norms(X_dist, squared=True)
    else:
        X_dist = X_work
        x_squared_norms_dist = x_squared_norms_work

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)


    n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.choice(
        n_samples, p=np.full(n_samples, 1.0 / n_samples)
    )
    indices = np.full(n_clusters, -1, dtype=int)

    centers[0] = X_work[center_id]
    indices[0] = index_map[center_id]

    # Initialize list of closest distances and calculate current potential
    first_center = X_dist[center_id]
    if hasattr(first_center, "ndim") and first_center.ndim == 1:
        first_center = first_center[np.newaxis]
    closest_dist_sq = _euclidean_distances(
        first_center, X_dist, Y_norm_squared=x_squared_norms_dist, squared=True
    )
    closest_dist_sq = closest_dist_sq.reshape(-1)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in trange(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            stable_cumsum(closest_dist_sq), rand_vals
        )
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            X_dist[candidate_ids],
            X_dist,
            Y_norm_squared=x_squared_norms_dist,
            squared=True,
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X_work[best_candidate]

        indices[c] = index_map[best_candidate]

    return centers, indices


###############################################################################
# K-means batch estimation by EM (expectation maximization)


def _tolerance(X, tol):
    """Return a tolerance which is dependent on the dataset."""
    if tol == 0:
        return 0

    variances = np.var(X, axis=0)
    return np.mean(variances) * tol


def _labels_inertia(X, centers, n_threads=1, return_inertia=True):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.

    Parameters
    ----------
    X : {ndarray} of shape (n_samples, n_features)
        The input samples to assign to the labels.

    x_squared_norms : ndarray of shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    return_inertia : bool, default=True
        Whether to compute and return the inertia.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The resulting assignment.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        Inertia is only returned if return_inertia is True.
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    labels = np.full(n_samples, -1, dtype=np.int32)
    center_shift = np.zeros(n_clusters, dtype=centers.dtype)


    _labels = lloyd_iter_chunked_dense
    _inertia = _inertia_dense

    sample_weight = np.ones(n_samples, dtype=X.dtype)

    _labels(
        X,
        sample_weight,
        centers,
        centers_new=None,
        weight_in_clusters=None,
        labels=labels,
        center_shift=center_shift,
        n_threads=n_threads,
        update_centers=False,
    )

    if return_inertia:
        inertia = _inertia(X, sample_weight, centers, labels, n_threads)
        return labels, inertia

    return labels


# Same as _labels_inertia but in a threadpool_limits context.
_labels_inertia_threadpool_limit = _threadpool_controller_decorator(
    limits=1, user_api="blas"
)(_labels_inertia)




def _mini_batch_step(
    X,
    centers,
    centers_new,
    weight_sums,
    random_state,
    random_reassign=False,
    reassignment_ratio=0.01,
    verbose=False,
    n_threads=1,
):
    """Incremental update of the centers for the Minibatch K-Means algorithm.

    Parameters
    ----------

    X : {ndarray} of shape (n_samples, n_features)
        The original data array.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared euclidean norm of each data point.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers before the current iteration

    centers_new : ndarray of shape (n_clusters, n_features)
        The cluster centers after the current iteration. Modified in-place.

    weight_sums : ndarray of shape (n_clusters,)
        The vector in which we keep track of the numbers of points in a
        cluster. This array is modified in place.

    random_state : RandomState instance
        Determines random number generation for low count centers reassignment.
        See :term:`Glossary <random_state>`.

    random_reassign : boolean, default=False
        If True, centers with very low counts are randomly reassigned
        to observations.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    verbose : bool, default=False
        Controls the verbosity.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation.

    Returns
    -------
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        The inertia is computed after finding the labels and before updating
        the centers.
    """
    # Perform label assignment to nearest centers
    # For better efficiency, it's better to run _mini_batch_step in a
    # threadpool_limit context than using _labels_inertia_threadpool_limit here
    labels, inertia = _labels_inertia(X, centers, n_threads=n_threads)


    sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    _minibatch_update_dense(
        X,
        sample_weight,
        centers,
        centers_new,
        weight_sums,
        labels,
        n_threads,
    )

    # Reassign clusters that have very low weight
    if random_reassign and reassignment_ratio > 0:
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()

        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > 0.5 * X.shape[0]:
            indices_dont_reassign = np.argsort(weight_sums)[int(0.5 * X.shape[0]) :]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()

        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = random_state.choice(
                X.shape[0], replace=False, size=n_reassigns
            )
            if verbose:
                print(f"[MiniBatchKMeans] Reassigning {n_reassigns} cluster centers.")

  
            centers_new[to_reassign] = X[new_centers]

        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])

    return inertia


class MiniBatchKMeans:
    """
    Mini-Batch K-Means clustering.

    Read more in the :ref:`User Guide <mini_batch_kmeans>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=1024
        Size of the mini batches.
        For faster computations, you can set the ``batch_size`` greater than
        256 * number of cores to enable parallelism on all cores.

        .. versionchanged:: 1.0
           `batch_size` default changed from 100 to 1024.

    verbose : int, default=0
        Verbosity mode.

    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.

        If `None`, the heuristic is `init_size = 3 * batch_size` if
        `3 * batch_size < n_clusters`, else `init_size = 3 * n_clusters`.

    n_init : 'auto' or int, default="auto"
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the best of
        the `n_init` initializations as measured by inertia. Several runs are
        recommended for sparse high-dimensional problems (see
        :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        3 if using `init='random'`;
        1 if using `init='k-means++'`.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'` in version.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a center to
        be reassigned. A higher value means that low count centers are more
        easily reassigned, which means that the model will take longer to
        converge, but should converge in a better clustering. However, too high
        a value may cause convergence issues, especially with a small batch
        size.

    init_projection_dim : int or None, default=None
        If set, apply a sparse random projection to this dimension when running
        k-means++ initialization for speed. Distances are computed in the
        projected space, while centers are chosen from the original data.


    init_subsample : int or None, default=None
        If set and smaller than the dataset size, subsample this many points
        for k-means++ initialization.

    Attributes
    ----------

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point (if compute_labels is set to True).

    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition if compute_labels is set to True. If compute_labels is set to
        False, it's an approximation of the inertia based on an exponentially
        weighted average of the batch inertiae.
        The inertia is defined as the sum of square distances of samples to
        their cluster center.

    n_iter_ : int
        Number of iterations over the full dataset.

    n_steps_ : int
        Number of minibatches processed.

        .. versionadded:: 1.0

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    KMeans : The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.

    Notes
    -----
    See https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

    When there are too few points in the dataset, some centers may be
    duplicated, which means that a proper clustering in terms of the number
    of requesting clusters and the number of returned clusters will not
    always match. One solution is to set `reassignment_ratio=0`, which
    prevents reassignments of clusters that are too small.

    Examples
    --------
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 0], [4, 4],
    ...               [4, 5], [0, 1], [2, 2],
    ...               [3, 2], [5, 5], [1, -1]])
    >>> # manually fit on batches
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          n_init="auto")
    >>> kmeans = kmeans.partial_fit(X[0:6,:])
    >>> kmeans = kmeans.partial_fit(X[6:12,:])
    >>> kmeans.cluster_centers_
    array([[3.375, 3.  ],
           [0.75 , 0.5 ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)
    >>> # fit on the whole data
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          max_iter=10,
    ...                          n_init="auto").fit(X)
    >>> kmeans.cluster_centers_
    array([[3.55102041, 2.48979592],
           [1.06896552, 1.        ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)
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
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.compute_labels = compute_labels
        self.random_state = random_state
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.init_size = init_size
        self.n_init = n_init
        self.reassignment_ratio = reassignment_ratio
        self.init_projection_dim = init_projection_dim
        self.init_subsample = init_subsample

    def _check_params_vs_input(self, X, default_n_init=3):
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )
        self._tol = _tolerance(X, self.tol)

        if not (isinstance(self.init, str) and self.init in {"k-means++", "random"}):
            raise ValueError(
                "init must be 'k-means++' or 'random', got "
                f"{self.init!r} instead."
            )
        if not (isinstance(self.init, str) and self.init in {"k-means++", "random"}):
            raise ValueError(
                "init must be 'k-means++' or 'random', got "
                f"{self.init!r} instead."
            )
        if self.n_init == "auto":
            if self.init == "k-means++":
                self._n_init = 1
            elif self.init == "random":
                self._n_init = default_n_init
            else:
                self._n_init = 1
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
                (
                    f"init_size={self._init_size} should be larger than "
                    f"n_clusters={self.n_clusters}. Setting it to "
                    "min(3*n_clusters, n_samples)"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])
        if self.reassignment_ratio < 0:
            raise ValueError(
                "reassignment_ratio should be >= 0, got "
                f"{self.reassignment_ratio} instead."
            )
        if self.init_projection_dim is not None and self.init_projection_dim <= 0:
            raise ValueError(
                "init_projection_dim should be > 0, got "
                f"{self.init_projection_dim} instead."
            )
        if self.init_subsample is not None and self.init_subsample < self.n_clusters:
            raise ValueError(
                "init_subsample should be >= n_clusters, got "
                f"{self.init_subsample} instead."
            )
        if self.init_projection_dim is not None and self.init_projection_dim <= 0:
            raise ValueError(
                "init_projection_dim should be > 0, got "
                f"{self.init_projection_dim} instead."
            )
        if self.init_subsample is not None and self.init_subsample < self.n_clusters:
            raise ValueError(
                "init_subsample should be >= n_clusters, got "
                f"{self.init_subsample} instead."
            )

    def _validate_center_shape(self, X, centers):
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    def _init_centroids(
        self,
        X,
        x_squared_norms,
        init,
        random_state,
        init_size=None,
        n_centroids=None,
    ):
        n_samples = X.shape[0]
        n_clusters = self.n_clusters if n_centroids is None else n_centroids
        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]
        if init == "k-means++":
            centers, _ = _kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                projection_dim=self.init_projection_dim,
                subsample=self.init_subsample,
            )
        elif init == "random":
            seeds = random_state.choice(
                n_samples,
                size=n_clusters,
                replace=False,
            )
            centers = X[seeds]
        else:
            raise ValueError(
                "init must be 'k-means++' or 'random', got "
                f"{init!r} instead."
            )

        return centers

    def _check_test_data(self, X):
        X = check_array(
            X,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return X

    def fit_predict(self, X):
        return self.fit(X).labels_

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        X = self._check_test_data(X)
        labels = _labels_inertia_threadpool_limit(
            X,
            self.cluster_centers_,
            n_threads=self._n_threads,
            return_inertia=False,
        )
        return labels

    def fit_transform(self, X):
        return self.fit(X)._transform(X)

    def transform(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        return euclidean_distances(X, self.cluster_centers_)

    def score(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model has not been fitted yet.")
        X = self._check_test_data(X)
        _, scores = _labels_inertia_threadpool_limit(
            X, self.cluster_centers_, self._n_threads
        )
        return -scores

    def _check_params_vs_input(self, X, default_n_init=3):
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        self._tol = _tolerance(X, self.tol)

        if self.n_init == "auto":
            if self.init == "k-means++":
                self._n_init = 1
            elif self.init == "random":
                self._n_init = default_n_init
            else:
                self._n_init = 1
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
                (
                    f"init_size={self._init_size} should be larger than "
                    f"n_clusters={self.n_clusters}. Setting it to "
                    "min(3*n_clusters, n_samples)"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])

        if self.reassignment_ratio < 0:
            raise ValueError(
                "reassignment_ratio should be >= 0, got "
                f"{self.reassignment_ratio} instead."
            )
    
    def _mini_batch_convergence(
        self, step, n_steps, n_samples, centers_squared_diff, batch_inertia
    ):
        """Helper function to encapsulate the early stopping logic"""
        # Normalize inertia to be able to compare values when
        # batch_size changes
        batch_inertia /= self._batch_size

        # count steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because it's inertia from initialization.
        if step == 1:
            if self.verbose:
                print(
                    f"Minibatch step {step}/{n_steps}: mean batch "
                    f"inertia: {batch_inertia}"
                )
            return False

        # Compute an Exponentially Weighted Average of the inertia to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_inertia is None:
            self._ewa_inertia = batch_inertia
        else:
            alpha = self._batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_inertia = self._ewa_inertia * (1 - alpha) + batch_inertia * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch inertia: "
                f"{batch_inertia}, ewa inertia: {self._ewa_inertia}"
            )

        # Early stopping based on absolute tolerance on squared change of
        # centers position
        if self._tol > 0.0 and centers_squared_diff <= self._tol:
            if self.verbose:
                print(f"Converged (small centers change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # inertia
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
                    "Converged (lack of improvement in inertia) at step "
                    f"{step}/{n_steps}"
                )
            return True

        return False

    def _random_reassign(self):
        """Check if a random reassignment needs to be done.

        Do random reassignments each time 10 * n_clusters samples have been
        processed.

        If there are empty clusters we always want to reassign.
        """
        self._n_since_last_reassign += self._batch_size
        if (self._counts == 0).any() or self._n_since_last_reassign >= (
            10 * self.n_clusters
        ):
            self._n_since_last_reassign = 0
            return True
        return False

    def fit(self, X):
        """Compute the centroids on X by chunking it into mini-batches.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(
            X,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        self._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        self._n_threads = _openmp_effective_n_threads()
        n_samples, n_features = X.shape

        init = self.init

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # Validation set for the init
        validation_indices = random_state.randint(0, n_samples, self._init_size)
        X_valid = X[validation_indices]
        # perform several inits with random subsets
        best_inertia = None
        total_init_time = 0.0
        for init_idx in range(self._n_init):
            if self.verbose:
                print(f"Init {init_idx + 1}/{self._n_init} with method {init}")

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans.
            init_start = time.perf_counter()
            cluster_centers = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                init_size=self._init_size,
            )
            init_elapsed = time.perf_counter() - init_start
            total_init_time += init_elapsed
            if self.verbose:
                print(
                    f"Init {init_idx + 1}/{self._n_init} centroids time: {init_elapsed:.6f}s"
                )

            # Compute inertia on a validation set.
            _, inertia = _labels_inertia_threadpool_limit(
                X_valid,
                cluster_centers,
                n_threads=self._n_threads,
            )

            if self.verbose:
                print(f"Inertia for init {init_idx + 1}/{self._n_init}: {inertia}")
            if best_inertia is None or inertia < best_inertia:
                init_centers = cluster_centers
                best_inertia = inertia

        centers = init_centers
        if self.verbose:
            print(f"Total centroid initialization time: {total_init_time:.6f}s")
        centers_new = np.empty_like(centers)

        # Initialize counts
        self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

        # Attributes to monitor the convergence
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0

        # Initialize number of samples seen since last reassignment
        self._n_since_last_reassign = 0

        n_steps = (self.max_iter * n_samples) // self._batch_size

        with _get_threadpool_controller().limit(limits=1, user_api="blas"):
            # Perform the iterative optimization until convergence
            for i in range(n_steps):
                # Sample a minibatch from the full dataset
                minibatch_indices = random_state.randint(0, n_samples, self._batch_size)

                # Perform the actual update step on the minibatch data
                batch_inertia = _mini_batch_step(
                    X=X[minibatch_indices],
                    centers=centers,
                    centers_new=centers_new,
                    weight_sums=self._counts,
                    random_state=random_state,
                    random_reassign=self._random_reassign(),
                    reassignment_ratio=self.reassignment_ratio,
                    verbose=self.verbose,
                    n_threads=self._n_threads,
                )

                if self._tol > 0.0:
                    centers_squared_diff = np.sum((centers_new - centers) ** 2)
                else:
                    centers_squared_diff = 0

                centers, centers_new = centers_new, centers

                # Monitor convergence and do early stopping if necessary
                if self._mini_batch_convergence(
                    i, n_steps, n_samples, centers_squared_diff, batch_inertia
                ):
                    break

        self.cluster_centers_ = centers
        self._n_features_out = self.cluster_centers_.shape[0]

        self.n_steps_ = i + 1
        self.n_iter_ = int(np.ceil(((i + 1) * self._batch_size) / n_samples))

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )
        else:
            self.inertia_ = self._ewa_inertia * n_samples

        return self