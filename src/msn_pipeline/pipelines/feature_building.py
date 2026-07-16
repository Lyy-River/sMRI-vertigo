from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def load_global_topology_features(
    health_topology_csv: Path,
    patient_topology_csv: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load global topology features for health and patient groups and concatenate.

    Returns
    -------
    subject_ids : (N,) array of str
    group : (N,) array of int (0=health, 1=patient)
    features : (N, D) float array with topology metrics only
    """
    h_df = pd.read_csv(health_topology_csv)
    p_df = pd.read_csv(patient_topology_csv)

    df = pd.concat([h_df, p_df], ignore_index=True)
    if "subject_id" not in df.columns or "group" not in df.columns:
        raise ValueError("Topology CSV 必须包含 subject_id 和 group 列。")

    subject_ids = df["subject_id"].astype(str).to_numpy()
    groups = df["group"].to_numpy(dtype=np.int64)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "group")]
    features = df[feature_cols].to_numpy(dtype=float)
    return subject_ids, groups, features


def load_gcn_features_csv(
    gcn_feature_csv: Path,
    id_col: str = "Patient_ID",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load precomputed GCN features from a CSV created by extract_all_patients.

    Returns
    -------
    subject_ids : (N,) array of str
    features : (N, D) float array
    """
    df = pd.read_csv(gcn_feature_csv)
    if id_col not in df.columns:
        raise ValueError(f"GCN 特征表中未找到 ID 列 {id_col!r}")
    subject_ids = df[id_col].astype(str).to_numpy()
    feature_cols = [c for c in df.columns if c != id_col]
    features = df[feature_cols].to_numpy(dtype=float)
    return subject_ids, features


def merge_features(
    subject_ids: Iterable[str],
    *feature_blocks: Iterable[np.ndarray],
) -> np.ndarray:
    """
    Horizontally concatenate multiple feature blocks for the same subject order.

    Assumes that all feature blocks are already aligned to the same subject_ids
    order as provided.
    """
    blocks = [np.asarray(b, dtype=float) for b in feature_blocks if b is not None]
    if not blocks:
        raise ValueError("No feature blocks provided for merge_features.")
    return np.hstack(blocks)


