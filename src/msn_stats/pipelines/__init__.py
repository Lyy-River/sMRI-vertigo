from __future__ import annotations

"""
High-level statistical pipelines for MSN analyses.

This package provides ready-to-use workflows for:
- NC vs patient NBS
- Cluster-level NBS / edgewise statistics
- Global / nodal topology GLM / ANCOVA
"""

from .nc_vs_patient_nbs import run_nc_vs_patient_nbs  # noqa: F401

__all__ = [
    "run_nc_vs_patient_nbs",
]

