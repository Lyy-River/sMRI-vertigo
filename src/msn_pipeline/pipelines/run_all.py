from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from msn_pipeline.config import PipelineConfig
from .msn_construction import build_group_msns
from .clustering_pipeline import run_clustering_with_topology_and_gcn


def run_full_pipeline(
    config: PipelineConfig,
    gcn_feature_csv: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Optional end-to-end pipeline:
    - Build health & patient MSNs and topology tables (if not already built)
    - Run clustering using topology (and optional GCN) features.
    """
    data_root = config.data_root
    results_root = config.output_root

    # MSN construction for both groups
    health_df = build_group_msns(
        group="health",
        data_root=data_root,
        results_root=results_root,
        metric=config.msn_metric,
        target_density=config.msn_target_density,
        compute_sigma=False,
        save_nodal=False,
    )
    patient_df = build_group_msns(
        group="patient",
        data_root=data_root,
        results_root=results_root,
        metric=config.msn_metric,
        target_density=config.msn_target_density,
        compute_sigma=False,
        save_nodal=False,
    )

    health_topology_csv = results_root / "health_subject_topology.csv"
    patient_topology_csv = results_root / "patient_subject_topology.csv"

    clustering_out_csv = (
        results_root / config.experiment_name / "subject_features_and_clusters.csv"
    )

    cluster_result = run_clustering_with_topology_and_gcn(
        health_topology_csv=health_topology_csv,
        patient_topology_csv=patient_topology_csv,
        gcn_feature_csv=gcn_feature_csv,
        method=config.clustering_method,
        n_clusters=config.clustering_n_clusters,
        out_csv=clustering_out_csv,
    )

    return {
        "health_topology": health_df,
        "patient_topology": patient_df,
        "clustering": cluster_result,
    }

