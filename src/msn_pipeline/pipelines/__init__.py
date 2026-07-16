from __future__ import annotations

"""
High-level MSN pipeline entry points.

This package groups together:
- MSN construction for health / patient groups
- Feature building (topology / GCN / other MSN-derived features)
- Clustering pipelines

Notebooks should import functions from here instead of re-implementing
loops and file I/O.
"""

from .msn_construction import build_group_msns  # noqa: F401
from .clustering_pipeline import (  # noqa: F401
    run_clustering_with_topology_and_gcn,
    run_patient_clustering_with_stability,
)

__all__ = [
    "build_group_msns",
    "run_clustering_with_topology_and_gcn",
    "run_patient_clustering_with_stability",
]

