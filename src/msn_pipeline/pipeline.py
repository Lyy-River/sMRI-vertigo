from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import PipelineConfig, load_config
from .data import DataPaths, load_subject_features
from .msn import build_msn
from .features import calculate_graph_metrics
from .clustering import cluster_subjects


def run_pipeline(config: PipelineConfig) -> Dict[str, object]:
    """
    High-level end-to-end pipeline:
    - Load subject features
    - Build MSN per subject
    - Extract global graph metrics
    - Cluster subjects
    """
    data_paths = DataPaths(root=config.data_root)
    patient_dir = data_paths.patient_csv_dir()
    subjects = load_subject_features(patient_dir)

    subject_ids = []
    feature_rows = []

    for sid, df in subjects.items():
        sim_matrix, G, regions = build_msn(
            df,
            metric=config.msn_metric,
            target_density=config.msn_target_density,
        )
        metrics = calculate_graph_metrics(
            G,
            metrics=["Eg", "Eloc", "Cp", "Lp", "σ"],
        )
        feature_rows.append(list(metrics.values()))
        subject_ids.append(sid)

    features = np.asarray(feature_rows, dtype=float)
    labels, diag = cluster_subjects(
        features,
        method=config.clustering_method,
        n_clusters=config.clustering_n_clusters,
    )

    results_dir = config.output_root / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(features)
    out_df.insert(0, "subject_id", subject_ids)
    out_df["cluster_label"] = labels
    out_path = results_dir / "subject_features_and_clusters.csv"
    out_df.to_csv(out_path, index=False)

    return {
        "subject_ids": subject_ids,
        "features": features,
        "labels": labels,
        "cluster_diagnostics": diag,
        "output_csv": out_path,
    }


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Run MSN pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    cli_main()
