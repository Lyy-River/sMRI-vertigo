"""Public NBS API: run_between_group_stats and re-exports from nbs_impl."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .nbs_impl import (
    NBSResult,
    component_mask_to_matrix,
    edges_to_matrix,
    matrix_to_edges,
    nbs_glm,
    nbs_glm_from_matrices,
)


def run_between_group_stats(
    msn_matrices: np.ndarray,
    design_matrix: np.ndarray,
    contrast: np.ndarray,
    n_nodes: int,
    primary_threshold: float,
    n_perm: int = 5000,
    two_tailed: bool = True,
    seed: Optional[int] = None,
) -> NBSResult:
    """
    Run GLM-based NBS for between-group comparison.

    msn_matrices: (n_subjects, n_nodes, n_nodes) similarity/adjacency matrices.
    design_matrix: (n_subjects, n_predictors), include intercept.
    contrast: (n_predictors,) e.g. [0, 1, 0] for group effect.
    n_nodes: number of nodes (e.g. 148).
    primary_threshold: t-statistic threshold for forming components.
    """
    return nbs_glm_from_matrices(
        matrices=msn_matrices,
        design=design_matrix,
        contrast=contrast,
        n_nodes=n_nodes,
        threshold=primary_threshold,
        n_perm=n_perm,
        two_tailed=two_tailed,
        seed=seed,
    )


__all__ = [
    "run_between_group_stats",
    "NBSResult",
    "nbs_glm",
    "nbs_glm_from_matrices",
    "matrix_to_edges",
    "edges_to_matrix",
    "component_mask_to_matrix",
]
