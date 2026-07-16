from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .similarity import get_similarity_function, compute_similarity_matrix


def build_similarity_matrix(
    features: np.ndarray,
    metric: str = "pearson",
    **kwargs: object,
) -> np.ndarray:
    """Compute region-by-region similarity matrix from feature matrix (no preprocessing)."""
    sim_func = get_similarity_function(metric)
    return compute_similarity_matrix(features, sim_func)


def create_network(
    regions: List[str],
    sim_matrix: np.ndarray,
    threshold: Optional[float] = None,
    target_density: float = 0.2,
) -> nx.Graph:
    """Build weighted graph from similarity matrix using threshold or target density."""
    G = nx.Graph()
    G.add_nodes_from(regions)
    if threshold is None:
        triu = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        threshold = float(np.percentile(triu, 100 * (1 - target_density)))
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if sim_matrix[i, j] >= threshold:
                G.add_edge(regions[i], regions[j], weight=float(sim_matrix[i, j]))
    return G


def build_msn(
    df: pd.DataFrame,
    metric: str = "pearson",
    threshold: Optional[float] = None,
    target_density: float = 0.2,
) -> Tuple[np.ndarray, nx.Graph, List[str]]:
    """Build MSN graph and similarity matrix for a single subject."""
    brain_volume = df.loc["BrainSegVolNotVent"].values[0] if "BrainSegVolNotVent" in df.index else 1.0
    if "volume" in df.columns:
        df = df.copy()
        df["volume"] = df["volume"] / brain_volume
    df = df.drop(index=["BrainSegVolNotVent", "eTIV"], errors="ignore")

    df_z = (df - df.mean()) / df.std()
    features = df_z.values

    sim_func = get_similarity_function(metric)
    sim_matrix = compute_similarity_matrix(features, sim_func)
    regions = df.index.tolist()

    G = nx.Graph()
    G.add_nodes_from(regions)

    if threshold is None:
        threshold = np.percentile(
            sim_matrix[np.triu_indices_from(sim_matrix, k=1)],
            100 * (1 - target_density),
        )

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if sim_matrix[i, j] >= threshold:
                G.add_edge(regions[i], regions[j], weight=sim_matrix[i, j])

    return sim_matrix, G, regions

