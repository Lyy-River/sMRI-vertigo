"""High-level API: build MSN and compute graph metrics in one call."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import networkx as nx

from .network_builder import build_msn


def calculate_brain_network(
    df: pd.DataFrame,
    similarity_metric: Union[str, Any] = "pearson",
    threshold: Optional[float] = None,
    target_density: float = 0.2,
    metrics: Optional[List[str]] = None,
    n_random: int = 50,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Build MSN from region-by-feature DataFrame and compute graph-level metrics.

    Uses the same preprocessing as build_msn (volume normalization, drop global rows, z-score).
    Returns dict with keys: graph, similarity_matrix (DataFrame), metrics (dict of metric name -> float).
    """
    if metrics is None:
        metrics = ["Eg", "Eloc", "Cp", "Lp", "γ", "λ", "σ"]

    sim_matrix, G, regions = build_msn(
        df,
        metric=similarity_metric if isinstance(similarity_metric, str) else "pearson",
        threshold=threshold,
        target_density=target_density,
    )

    from msn_pipeline.features.topology import calculate_graph_metrics

    metrics_dict = calculate_graph_metrics(G, metrics, n_random=n_random)

    sim_df = pd.DataFrame(sim_matrix, index=regions, columns=regions)

    return {
        "graph": G,
        "similarity_matrix": sim_df,
        "metrics": metrics_dict,
    }
