from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from msn_pipeline.clustering import cluster_subjects
from msn_pipeline.clustering.stability import bootstrap_ari_stability
from .feature_building import (
    load_gcn_features_csv,
    merge_features,
)


def _load_patient_topology_features(
    patient_topology_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    仅加载患者的全局拓扑特征。

    返回
    -----
    subject_ids : (N,) str
    groups      : (N,) int (全部应为 1)
    features    : (N, D) float
    """
    df = pd.read_csv(patient_topology_csv)
    if "subject_id" not in df.columns or "group" not in df.columns:
        raise ValueError("patient_subject_topology.csv 必须包含 subject_id 和 group 列。")

    subject_ids = df["subject_id"].astype(str).to_numpy()
    groups = df["group"].to_numpy(dtype=np.int64)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "group")]
    features = df[feature_cols].to_numpy(dtype=float)
    return subject_ids, groups, features


def run_clustering_with_topology_and_gcn(
    patient_topology_csv: Path,
    gcn_feature_csv: Optional[Path] = None,
    method: str = "kmeans",
    n_clusters: int = 3,
    out_csv: Optional[Path] = None,
) -> Dict[str, object]:
    """
    患者内部聚类的高层 pipeline：
    1. 从 patient_subject_topology.csv 读取全局拓扑特征
    2. 可选：读取患者 GCN 特征 CSV，并与拓扑特征拼接
    3. 调用 clustering.strategies.cluster_subjects 进行聚类
    4. （可选）保存聚类结果到 CSV

    返回
    -----
    dict，包含：subject_ids, group, features, labels, diagnostics, output_csv
    """
    subject_ids, groups, topo_features = _load_patient_topology_features(
        patient_topology_csv
    )

    features = topo_features

    if gcn_feature_csv is not None:
        gcn_ids, gcn_features = load_gcn_features_csv(gcn_feature_csv)
        # 对齐顺序：假定 gcn_feature_csv 只包含 patient，且 ID 与 subject_ids 子集一致
        id_to_idx = {sid: i for i, sid in enumerate(gcn_ids)}
        gcn_rows = []
        for sid in subject_ids:
            if sid in id_to_idx:
                gcn_rows.append(gcn_features[id_to_idx[sid]])
            else:
                # 若没有对应 GCN 特征，用 0 向量占位
                gcn_rows.append(np.zeros(gcn_features.shape[1], dtype=float))
        gcn_block = np.vstack(gcn_rows)
        features = merge_features(subject_ids, topo_features, gcn_block)

    labels, diag = cluster_subjects(
        features,
        method=method,
        n_clusters=n_clusters,
    )

    output_csv: Optional[Path] = None
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(features)
        df.insert(0, "group", groups)
        df.insert(0, "SubjectID", subject_ids)
        df["Cluster"] = labels
        df.to_csv(out_csv, index=False)
        output_csv = out_csv

    return {
        "subject_ids": subject_ids,
        "group": groups,
        "features": features,
        "labels": labels,
        "diagnostics": diag,
        "output_csv": output_csv,
    }


def run_patient_clustering_with_stability(
    patient_topology_csv: Path,
    gcn_feature_csv: Optional[Path] = None,
    method: str = "kmeans",
    n_clusters: int = 3,
    n_bootstrap: int = 100,
    sample_ratio: float = 0.8,
    out_csv: Optional[Path] = None,
    random_state: int = 0,
) -> Dict[str, object]:
    """
    在患者内部聚类的基础上，使用 bootstrap + pairwise ARI 评估聚类稳定性。

    返回结果在 run_clustering_with_topology_and_gcn 的 dict 基础上，
    额外包含一个 \"stability\" 字段。
    """
    base = run_clustering_with_topology_and_gcn(
        patient_topology_csv=patient_topology_csv,
        gcn_feature_csv=gcn_feature_csv,
        method=method,
        n_clusters=n_clusters,
        out_csv=out_csv,
    )

    features = base["features"]

    def _cluster_fn(X: np.ndarray) -> np.ndarray:
        labs, _ = cluster_subjects(
            X,
            method=method,
            n_clusters=n_clusters,
        )
        return labs

    stability = bootstrap_ari_stability(
        features=features,
        cluster_fn=_cluster_fn,
        n_bootstrap=n_bootstrap,
        sample_ratio=sample_ratio,
        random_state=random_state,
    )

    base["stability"] = stability
    return base

