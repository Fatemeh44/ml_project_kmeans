
import numpy as np

from .kmeans import KMeansScratch, squared_euclidean_distances


def initialize_kmeans_pp(X, n_clusters, random_state=None):
    """
    Initialize cluster centers using the k-means++ strategy.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Dataset.
    n_clusters : int
        Number of clusters.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    centers : ndarray, shape (n_clusters, n_features)
        Initial centers selected by k-means++.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n_samples, n_features = X.shape

    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")

    if n_clusters > n_samples:
        raise ValueError("n_clusters cannot be larger than the number of samples.")

    rng = np.random.default_rng(random_state)

    centers = np.empty((n_clusters, n_features), dtype=float)

    # Step 1: choose the first center uniformly at random
    first_index = rng.integers(n_samples)
    centers[0] = X[first_index]

    selected_indices = {int(first_index)}

    # Steps 2-4: choose each next center using D^2 sampling
    for center_id in range(1, n_clusters):
        distances = squared_euclidean_distances(X, centers[:center_id])
        closest_squared_distances = np.min(distances, axis=1)

        # Avoid selecting the exact same point again
        for idx in selected_indices:
            closest_squared_distances[idx] = 0.0

        total_distance = np.sum(closest_squared_distances)

        if total_distance == 0.0:
            # Degenerate case: all remaining points are identical to selected centers.
            remaining_indices = np.array(
                [idx for idx in range(n_samples) if idx not in selected_indices]
            )
            next_index = rng.choice(remaining_indices)
        else:
            probabilities = closest_squared_distances / total_distance
            next_index = rng.choice(n_samples, p=probabilities)

        centers[center_id] = X[next_index]
        selected_indices.add(int(next_index))

    return centers


class KMeansPlusPlusScratch(KMeansScratch):
    """
    K-Means using k-means++ initialization, implemented from scratch.

    The initialization is done with D^2 sampling.
    The optimization step is the same standard K-Means algorithm.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )

        self.initial_centers_ = None

    def fit(self, X):
        """
        Fit K-Means with k-means++ initialization.
        """
        X = self._validate_data(X)

        self.initial_centers_ = initialize_kmeans_pp(
            X,
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )

        super().fit(X, initial_centroids=self.initial_centers_)

        return self

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.
        """
        self.fit(X)
        return self.labels_
