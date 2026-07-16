"""
High-level package for the MSN pipeline.

This package provides:
- Data loading utilities
- MSN construction and thresholding
- Feature extraction (graph metrics and GCN-based)
- Clustering strategies
- Statistical comparison helpers
- End-to-end pipeline entry points
"""

from .pipeline import run_pipeline  # convenience re-export

__all__ = ["run_pipeline"]

