from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from msn_pipeline.data import DataPaths, load_subject_features
from msn_pipeline.msn import build_msn
from msn_pipeline.features import calculate_graph_metrics, compute_nodal_topology


GroupType = Literal["health", "patient"]


def _get_dirs(
    group: GroupType,
    results_root: Path,
) -> tuple[Path, Path, Path]:
    """
    Return (msn_dir, nodal_dir, progress_csv) for the given group.
    """
    if group == "health":
        msn_dir = results_root / "health_msn_matrices"
        nodal_dir = results_root / "health_nodal_topology"
        progress_csv = results_root / "health_msn_progress.csv"
    else:
        msn_dir = results_root / "patient_msn_matrices"
        nodal_dir = results_root / "patient_nodal_topology"
        progress_csv = results_root / "patient_msn_progress.csv"

    msn_dir.mkdir(parents=True, exist_ok=True)
    nodal_dir.mkdir(parents=True, exist_ok=True)
    progress_csv.parent.mkdir(parents=True, exist_ok=True)
    return msn_dir, nodal_dir, progress_csv


def build_group_msns(
    group: GroupType,
    data_root: Path,
    results_root: Path,
    metric: str = "pearson",
    target_density: float = 0.2,
    compute_sigma: bool = False,
    save_nodal: bool = False,
    progress_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build MSN matrices and global topology metrics for one group.

    Parameters
    ----------
    group:
        "health" or "patient".
    data_root:
        Project data root (contains health_feature_matrices / patient_feature_matrices).
    results_root:
        Root directory for results (MSN matrices, nodal topology, summary CSV).
    metric:
        Similarity metric for MSN construction (e.g., "pearson").
    target_density:
        Target graph density for thresholding.
    compute_sigma:
        Whether to compute small-world sigma (can be expensive).
    save_nodal:
        Whether to save nodal topology CSV per subject.
    progress_csv:
        Optional custom progress CSV path. If None, a default under results_root
        is used per group.

    Returns
    -------
    pd.DataFrame
        Table with columns: subject_id, group, Eg, Eloc, Cp, Lp, (optional σ).
    """
    data_paths = DataPaths(root=data_root)
    if group == "health":
        feature_dir = data_paths.control_csv_dir()
        group_label = 0
    else:
        feature_dir = data_paths.patient_csv_dir()
        group_label = 1

    subjects = load_subject_features(feature_dir)

    msn_dir, nodal_dir, default_progress = _get_dirs(group, results_root)
    if progress_csv is None:
        progress_csv = default_progress

    metric_names: list[str] = ["Eg", "Eloc", "Cp", "Lp"]
    if compute_sigma:
        metric_names.append("σ")

    done_ids: set[str] = set()
    global_rows: list[dict[str, object]] = []

    if progress_csv.exists():
        prog_df = pd.read_csv(progress_csv)
        if not prog_df.empty and "subject_id" in prog_df.columns:
            global_rows = prog_df.to_dict("records")
            done_ids = set(prog_df["subject_id"].astype(str).tolist())

    for sid, df in subjects.items():
        if sid in done_ids:
            continue

        sim_matrix, G, regions = build_msn(
            df,
            metric=metric,
            target_density=target_density,
        )

        # Fisher Z transform for saving MSN matrix
        r_clip = np.clip(sim_matrix, -1 + 1e-8, 1 - 1e-8)
        z_matrix = np.arctanh(r_clip)
        np.fill_diagonal(z_matrix, 0.0)
        z_df = pd.DataFrame(z_matrix, index=regions, columns=regions)
        msn_path = msn_dir / f"{sid}_feature_matrix_msn.csv"
        z_df.to_csv(msn_path)

        metrics = calculate_graph_metrics(G, metrics=metric_names)
        row: dict[str, object] = {"subject_id": sid, "group": group_label}
        row.update(metrics)
        global_rows.append(row)
        done_ids.add(sid)

        if save_nodal:
            nodal_df = compute_nodal_topology(G)
            nodal_df.insert(0, "subject_id", sid)
            nodal_path = nodal_dir / f"{sid}_nodal_topology.csv"
            nodal_df.to_csv(nodal_path)

        pd.DataFrame(global_rows).to_csv(progress_csv, index=False)

    summary_df = pd.DataFrame(global_rows)
    if summary_df.empty:
        return summary_df

    # Optional: compute sigma afterwards from saved Z matrices if requested but
    # not included in the initial metric list.
    if compute_sigma and "σ" not in summary_df.columns:
        sigma_list: list[float] = []
        for row in global_rows:
            sid = str(row["subject_id"])
            msn_path = msn_dir / f"{sid}_feature_matrix_msn.csv"
            if not msn_path.exists():
                sigma_list.append(np.nan)
                continue
            z_df = pd.read_csv(msn_path, index_col=0)
            z_matrix = z_df.values.astype(float)
            r_matrix = np.tanh(z_matrix)
            regions = list(z_df.index)
            from msn_pipeline.msn import create_network

            G = create_network(regions, r_matrix, target_density=target_density)
            sigma_val = calculate_graph_metrics(G, metrics=["σ"]).get("σ", np.nan)
            sigma_list.append(float(sigma_val) if sigma_val is not None else np.nan)
        summary_df["σ"] = sigma_list

    # Standardize column order: subject_id, group, others
    cols = ["subject_id", "group"] + [
        c for c in summary_df.columns if c not in ("subject_id", "group")
    ]
    summary_df = summary_df[cols]

    out_name = f"{group}_subject_topology.csv"
    out_path = results_root / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)

    return summary_df

