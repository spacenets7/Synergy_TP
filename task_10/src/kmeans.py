"""
kmeans.py
---------
KMeans clustering implemented from scratch.
No scikit-learn. Only numpy.

Algorithm:
    1. Initialize k centroids using KMeans++ strategy.
    2. Assign each point to the nearest centroid (Euclidean distance).
    3. Update each centroid to the mean of its assigned points.
    4. Repeat 2-3 until assignments do not change or max_iters reached.
"""

import numpy as np


class KMeans:
    """
    KMeans clustering from scratch with KMeans++ initialization.

    Parameters
    ----------
    k : int
        Number of clusters.
    max_iters : int
        Maximum number of assignment-update iterations.
    random_seed : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        k:           int = 3,
        max_iters:   int = 300,
        random_seed: int = 42,
    ) -> None:
        self.k           = k
        self.max_iters   = max_iters
        self.random_seed = random_seed

        self.centroids_:    np.ndarray | None = None
        self.labels_:       np.ndarray | None = None
        self.inertia_:      float | None      = None
        self.n_iters_:      int = 0

    def _init_centroids_plusplus(self, X: np.ndarray) -> np.ndarray:
        """
        KMeans++ initialization.

        Choose first centroid uniformly at random.
        Each subsequent centroid is chosen with probability proportional
        to the squared distance from the nearest already-chosen centroid.
        This reduces the risk of poor initialization and speeds convergence.

        Args:
            X: Data matrix (n, p).

        Returns:
            Initial centroid matrix (k, p).
        """
        rng = np.random.default_rng(self.random_seed)
        n   = len(X)

        # First centroid: random point
        first_idx = rng.integers(0, n)
        centroids = [X[first_idx]]

        for _ in range(1, self.k):
            # Squared distances to nearest centroid
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            # Sample with probability ∝ distance²
            probs = dists / dists.sum()
            idx   = rng.choice(n, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each point to the nearest centroid (argmin Euclidean distance).

        Args:
            X:         Data matrix (n, p).
            centroids: Centroid matrix (k, p).

        Returns:
            Label array (n,) of cluster indices.
        """
        # (n, k) distance matrix
        dists  = np.array([
            np.sum((X - c) ** 2, axis=1)
            for c in centroids
        ]).T   # shape (n, k)
        return np.argmin(dists, axis=1)

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray, old_centroids: np.ndarray
    ) -> np.ndarray:
        """
        Update each centroid to the mean of its assigned points.
        If a cluster becomes empty, keep the old centroid.

        Args:
            X:             Data matrix (n, p).
            labels:        Assignment array (n,).
            old_centroids: Previous centroid matrix (k, p).

        Returns:
            Updated centroid matrix (k, p).
        """
        new_centroids = np.zeros_like(old_centroids)
        for i in range(self.k):
            mask = labels == i
            if mask.sum() > 0:
                new_centroids[i] = X[mask].mean(axis=0)
            else:
                new_centroids[i] = old_centroids[i]   # keep old centroid
        return new_centroids

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Run KMeans clustering.

        Args:
            X: Data matrix (n, p). Should be standardized.

        Returns:
            self with labels_, centroids_, inertia_, n_iters_ set.
        """
        centroids = self._init_centroids_plusplus(X)

        labels = np.zeros(len(X), dtype=int)

        for iteration in range(self.max_iters):
            new_labels = self._assign_labels(X, centroids)

            # Check convergence
            if np.array_equal(new_labels, labels) and iteration > 0:
                self.n_iters_ = iteration
                break

            labels    = new_labels
            centroids = self._update_centroids(X, labels, centroids)
        else:
            self.n_iters_ = self.max_iters

        self.labels_    = labels
        self.centroids_ = centroids

        # Compute inertia
        self.inertia_ = float(sum(
            np.sum((X[labels == i] - centroids[i]) ** 2)
            for i in range(self.k)
            if np.sum(labels == i) > 0
        ))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to the nearest fitted centroid."""
        if self.centroids_ is None:
            raise RuntimeError("KMeans has not been fitted.")
        return self._assign_labels(X, self.centroids_)
