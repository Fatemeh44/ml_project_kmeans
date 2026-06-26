
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def ensure_dir(directory):
    """
    Create a directory if it does not already exist.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def standardize_features(X):
    """
    Standardize features to zero mean and unit variance.

    This is important for distance-based algorithms like K-Means,
    because features with larger numerical scales can dominate distances.
    """
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Avoid division by zero for constant features
    std_safe = np.where(std == 0, 1.0, std)

    X_scaled = (X - mean) / std_safe

    return X_scaled, mean, std_safe


def save_figure(fig, save_path, dpi=300):
    """
    Save a matplotlib figure with tight layout.
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_2d_dataset(X, title, save_path, labels=None):
    """
    Plot a 2D dataset.

    Parameters
    ----------
    X : ndarray, shape (n_samples, 2)
        Two-dimensional dataset.
    title : str
        Plot title.
    save_path : str or Path
        Where to save the figure.
    labels : optional array-like
        If provided, points are colored by label.
    """
    X = np.asarray(X, dtype=float)

    if X.shape[1] != 2:
        raise ValueError("plot_2d_dataset requires X with exactly 2 features.")

    fig, ax = plt.subplots(figsize=(7, 5))

    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=30, alpha=0.8)
    else:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=30, alpha=0.8)
        fig.colorbar(scatter, ax=ax, label="Label")

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)

    save_figure(fig, save_path)


def plot_clusters_2d(X, labels, centers, title, save_path):
    """
    Plot 2D clustering results with centroids.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    centers = np.asarray(centers, dtype=float)

    if X.shape[1] != 2:
        raise ValueError("plot_clusters_2d requires X with exactly 2 features.")

    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        s=30,
        alpha=0.8
    )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=250,
        edgecolor="black",
        linewidth=1.5,
        label="Centroids"
    )

    fig.colorbar(scatter, ax=ax, label="Cluster label")
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, save_path)


def plot_metric_boxplot(values_random, values_pp, ylabel, title, save_path):
    """
    Create a boxplot comparing random initialization and k-means++.
    """
    values_random = np.asarray(values_random, dtype=float)
    values_pp = np.asarray(values_pp, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(
        [values_random, values_pp],
        labels=["Random", "K-Means++"],
        showmeans=True
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    save_figure(fig, save_path)


def plot_convergence_curves(history_random, history_pp, title, save_path):
    """
    Plot inertia over iterations for one random run and one k-means++ run.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        range(1, len(history_random) + 1),
        history_random,
        marker="o",
        label="Random initialization"
    )

    ax.plot(
        range(1, len(history_pp) + 1),
        history_pp,
        marker="o",
        label="K-Means++ initialization"
    )

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Inertia")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, save_path)


def summarize_runs(inertias, iterations):
    """
    Return a small dictionary with summary statistics for multiple runs.
    """
    inertias = np.asarray(inertias, dtype=float)
    iterations = np.asarray(iterations, dtype=float)

    return {
        "mean_inertia": float(np.mean(inertias)),
        "std_inertia": float(np.std(inertias)),
        "min_inertia": float(np.min(inertias)),
        "max_inertia": float(np.max(inertias)),
        "mean_iterations": float(np.mean(iterations)),
        "std_iterations": float(np.std(iterations)),
    }
