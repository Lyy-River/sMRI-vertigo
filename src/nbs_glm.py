"""
Deprecated shim for NBS functionality.

Use the implementations in ``msn_stats.nbs_glm`` instead.
This module is kept temporarily for backward compatibility and simply
re-exports the public API from ``msn_stats.nbs_glm``.
"""

from __future__ import annotations

from msn_stats.nbs_glm import (  # noqa: F401
    NBSResult,
    component_mask_to_matrix,
    edges_to_matrix,
    matrix_to_edges,
    nbs_glm,
    nbs_glm_from_matrices,
)

__all__ = [
    "NBSResult",
    "matrix_to_edges",
    "edges_to_matrix",
    "nbs_glm",
    "nbs_glm_from_matrices",
    "component_mask_to_matrix",
]

