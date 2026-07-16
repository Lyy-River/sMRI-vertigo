from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors


def discretize(x: np.ndarray, bins: int = 10) -> np.ndarray:
    """Discretize continuous values for mutual information."""
    return np.digitize(x, np.histogram_bin_edges(x, bins=bins))


def knn_kl_divergence(X: np.ndarray, Y: np.ndarray, k: int = 1) -> float:
    """Estimate KL(P || Q) where P is from X, Q from Y (k-NN estimator)."""
    n, d = X.shape
    nbrs_X = NearestNeighbors(n_neighbors=k + 1).fit(X)
    r = nbrs_X.kneighbors(X, return_distance=True)[0][:, k]
    nbrs_Y = NearestNeighbors(n_neighbors=k).fit(Y)
    s = nbrs_Y.kneighbors(X, return_distance=True)[0][:, k - 1]
    epsilon = 1e-10
    r, s = np.maximum(r, epsilon), np.maximum(s, epsilon)
    return float(d * np.mean(np.log(s / r)) + np.log(len(Y) / (len(X) - 1)))


def symmetric_kl_divergence(x: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Symmetric KL divergence; returns non-negative value (negative clamped to 0)."""
    X, Y = x.reshape(-1, 1), y.reshape(-1, 1)
    D_ij = knn_kl_divergence(X, Y, k)
    D_ji = knn_kl_divergence(Y, X, k)
    return max(0.0, 0.5 * (D_ij + D_ji))


def get_similarity_function(metric: Union[str, Callable]) -> Callable:
    """Return similarity function by name (wrapper around existing logic)."""
    metrics = {
        "pearson": lambda x, y: pearsonr(x, y)[0],
        "cosine": lambda x, y: 1 - cosine(x, y),
        "cross_corr": lambda x, y: np.correlate(x - x.mean(), y - y.mean())[0] / (
            len(x) * x.std() * y.std()
        ),
        "mutual_info": lambda x, y: mutual_info_regression(x.reshape(-1, 1), y)[0],
        "mutual_info_discrete": lambda x, y: mutual_info_score(discretize(x), discretize(y)),
        "kl_divergence": lambda x, y: 1.0 / (1.0 + symmetric_kl_divergence(x, y)),
    }

    if callable(metric):
        return metric
    if metric in metrics:
        return metrics[metric]
    raise ValueError(f"Unsupported metric: {metric}. Available: {list(metrics.keys())}")


def compute_similarity_matrix(data: np.ndarray, sim_func: Callable) -> np.ndarray:
    """Compute region-by-region similarity matrix."""
    n = data.shape[0]
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                val = sim_func(data[i], data[j])
                sim_matrix[i, j] = sim_matrix[j, i] = val
    return sim_matrix

