from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


def cluster_subjects(
    features: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 3,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Cluster subjects based on feature matrix.

    Parameters
    ----------
    features : np.ndarray
        Shape (n_subjects, n_features)

    method : str
        Clustering method:
        - kmeans
        - hierarchical
        - spectral
        - gmm
        - ward
        - average

    n_clusters : int
        Number of clusters

    normalize : bool
        Whether to z-score features

    Returns
    -------
    labels : np.ndarray
        Cluster labels

    metrics : Dict[str, float]
        Clustering quality metrics
    """

    X = features.copy()

    if normalize:
        X = StandardScaler().fit_transform(X)

    # ---------------------------
    # clustering model
    # ---------------------------

    if method == "kmeans":

        model = KMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init="auto",
        )

        labels = model.fit_predict(X)

        model_metrics = {
            "inertia": float(model.inertia_)
        }

    elif method == "hierarchical":

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="complete",
        )

        labels = model.fit_predict(X)

        model_metrics = {}

    elif method == "ward":

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        )

        labels = model.fit_predict(X)

        model_metrics = {}

    elif method == "average":

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="average",
        )

        labels = model.fit_predict(X)

        model_metrics = {}

    elif method == "spectral":

        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=10,
            random_state=0,
        )

        labels = model.fit_predict(X)

        model_metrics = {}

    elif method == "gmm":

        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=0,
        )

        labels = model.fit_predict(X)

        model_metrics = {
            "bic": float(model.bic(X)),
            "aic": float(model.aic(X)),
            "log_likelihood": float(model.score(X)),
        }

    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # ---------------------------
    # clustering quality metrics
    # ---------------------------

    metrics = model_metrics.copy()

    if len(np.unique(labels)) > 1:

        metrics.update(
            {
                "silhouette": float(silhouette_score(X, labels)),
                "davies_bouldin": float(davies_bouldin_score(X, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            }
        )

    return labels, metrics