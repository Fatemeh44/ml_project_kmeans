
import numpy as np


def squared_euclidean_distances(X, centers):
    """
    Compute squared Euclidean distances between each data point and each center.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Dataset.
    centers : array-like, shape (n_clusters, n_features)
        Cluster centers.

    Returns
    -------
    distances : ndarray, shape (n_samples, n_clusters)
        distances[i, j] is the squared distance between X[i] and centers[j].
    """
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)

    return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)


class KMeansScratch:
    """
    K-Means implemented from scratch using NumPy.

    This implementation does not use sklearn.fit().
    It supports random initialization and stores:
    - final cluster centers
    - final labels
    - final inertia
    - inertia history over iterations
    - number of iterations
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if tol < 0:
            raise ValueError("tol must be non-negative.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_history_ = []
        self.n_iter_ = 0

    def _validate_data(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        n_samples, n_features = X.shape

        if n_samples == 0:
            raise ValueError("X must contain at least one sample.")

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be larger than the number of samples.")

        return X

    def _initialize_random_centroids(self, X, rng):
        """
        Randomly select k different points from X as initial centroids.
        """
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _assign_labels(self, X, centroids):
        """
        Assign each point to the nearest centroid.
        """
        distances = squared_euclidean_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        return labels

    def _compute_inertia(self, X, centroids, labels):
        """
        Compute the K-Means objective:
        sum of squared distances from each point to its assigned centroid.
        """
        return float(np.sum((X - centroids[labels]) ** 2))

    def _update_centroids(self, X, labels, old_centroids, rng):
        """
        Update each centroid as the mean of the assigned points.

        If a cluster becomes empty, reinitialize that centroid using
        the point that is farthest from its nearest current centroid.
        This avoids crashes during difficult random initializations.
        """
        new_centroids = np.empty_like(old_centroids)

        distances_to_old = squared_euclidean_distances(X, old_centroids)
        nearest_old_distance = np.min(distances_to_old, axis=1)

        for cluster_id in range(self.n_clusters):
            points_in_cluster = X[labels == cluster_id]

            if len(points_in_cluster) > 0:
                new_centroids[cluster_id] = np.mean(points_in_cluster, axis=0)
            else:
                farthest_index = np.argmax(nearest_old_distance)
                new_centroids[cluster_id] = X[farthest_index]

        return new_centroids

    def fit(self, X, initial_centroids=None):
        """
        Fit K-Means to data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Dataset.
        initial_centroids : optional ndarray, shape (n_clusters, n_features)
            If provided, these centers are used instead of random initialization.
            This will be useful later for K-Means++.
        """
        X = self._validate_data(X)
        rng = np.random.default_rng(self.random_state)

        if initial_centroids is None:
            centroids = self._initialize_random_centroids(X, rng)
        else:
            centroids = np.asarray(initial_centroids, dtype=float)

            if centroids.shape != (self.n_clusters, X.shape[1]):
                raise ValueError(
                    "initial_centroids must have shape "
                    f"({self.n_clusters}, {X.shape[1]})."
                )

            centroids = centroids.copy()

        self.inertia_history_ = []

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_labels(X, centroids)
            inertia = self._compute_inertia(X, centroids, labels)
            self.inertia_history_.append(inertia)

            new_centroids = self._update_centroids(X, labels, centroids, rng)
            centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))

            centroids = new_centroids

            if centroid_shift <= self.tol:
                break

        final_labels = self._assign_labels(X, centroids)
        final_inertia = self._compute_inertia(X, centroids, final_labels)

        if len(self.inertia_history_) == 0 or not np.isclose(self.inertia_history_[-1], final_inertia):
            self.inertia_history_.append(final_inertia)

        self.cluster_centers_ = centroids
        self.labels_ = final_labels
        self.inertia_ = final_inertia
        self.n_iter_ = len(self.inertia_history_)

        return self

    def predict(self, X):
        """
        Assign new points to the nearest fitted centroid.
        """
        if self.cluster_centers_ is None:
            raise ValueError("This KMeansScratch instance is not fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X, initial_centroids=None):
        """
        Fit the model and return cluster labels.
        """
        self.fit(X, initial_centroids=initial_centroids)
        return self.labels_
