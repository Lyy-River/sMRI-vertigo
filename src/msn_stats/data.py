"""
MSN 网络分析 — 数据准备。

- 从 results/health_msn_matrices、results/patient_msn_matrices 加载 148×148 MSN 矩阵
- 从 data/*_info_processed.csv 合并 age, sex, eTIV（无 processed 时回退到 *_info.csv 的 age, sex）
- 向量化为 N × 10878 边矩阵
- 构建设计矩阵：两组 [1, group, sex, age_centered, eTIV_centered]；多组 [1, d1, d2, sex, age_centered, eTIV_centered]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


N_NODES = 148
N_EDGES = N_NODES * (N_NODES - 1) // 2  # 10878


def _n_edges(n: int) -> int:
    return n * (n - 1) // 2


def matrix_to_edges(A: np.ndarray) -> np.ndarray:
    """(n, n) -> (n_edges,) 或 (n_subj, n, n) -> (n_subj, n_edges)。上三角行优先。"""
    if A.ndim == 2:
        n = A.shape[0]
        return A[np.triu_indices(n, k=1)]
    if A.ndim == 3:
        n_subj, n, _ = A.shape
        E = _n_edges(n)
        out = np.empty((n_subj, E), dtype=A.dtype)
        for i in range(n_subj):
            out[i] = A[i][np.triu_indices(n, k=1)]
        return out
    raise ValueError("A must be 2D or 3D")


def load_msn_csv(path: Path, n_nodes: int = N_NODES) -> np.ndarray:
    """
    读取 MSN CSV（148×148）。支持：带行列标签的表格，或纯数值矩阵。
    """
    try:
        df = pd.read_csv(path, index_col=0)
        mat = df.iloc[:n_nodes, :n_nodes].values.astype(np.float64)
    except Exception:
        df = pd.read_csv(path, header=None)
        mat = df.iloc[:n_nodes, :n_nodes].values.astype(np.float64)
    if mat.shape[0] != n_nodes or mat.shape[1] != n_nodes:
        raw = pd.read_csv(path, header=None).values.astype(np.float64)
        if raw.shape[0] >= n_nodes and raw.shape[1] >= n_nodes:
            mat = raw[:n_nodes, :n_nodes].astype(np.float64)
        else:
            raise ValueError(f"MSN 矩阵行/列不足 {n_nodes}: {path}")
    return mat


def _find_msn_path(
    subject_id: str,
    msn_dir: Path,
    n_nodes: int = N_NODES,
) -> Optional[Path]:
    """根据 subject_id 查找 MSN 文件，仅读取后缀为 _feature_matrix_msn.csv 的文件。"""
    p = msn_dir / f"{subject_id}_feature_matrix_msn.csv"
    return p if p.exists() else None


def load_nc_patient_data(
    health_msn_dir: Path,
    patient_msn_dir: Path,
    health_info_path: Path,
    patient_info_path: Path,
    n_nodes: int = N_NODES,
    use_processed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    加载 NC + Patient 的 MSN 与协变量，仅保留在 info 中有记录且存在 MSN 文件的被试。

    Returns
    -------
    edge_matrix : (N, n_edges) float
    group : (N,) 0=NC, 1=Patient
    age : (N,) 岁
    sex : (N,) 0=f, 1=m
    eTIV : (N,) 或全 1（无 eTIV 时）
    subject_ids : list of str
    """
    # 优先使用带 eTIV 的 processed 表
    if use_processed:
        health_info_p = health_info_path.parent / (health_info_path.stem.replace("info", "info_processed") + ".csv")
        patient_info_p = patient_info_path.parent / (patient_info_path.stem.replace("info", "info_processed") + ".csv")
        if health_info_p.exists():
            health_info_path = health_info_p
        if patient_info_p.exists():
            patient_info_path = patient_info_p

    def _read_info(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p, dtype={"ID": str, "Age": np.float64})
        df["ID"] = df["ID"].astype(str).str.strip()
        if "Sex" in df.columns:
            df["sex"] = df["Sex"]
        elif "Gender" in df.columns:
            df["sex"] = df["Gender"].map({"f": 0, "m": 1})
        if "Age" in df.columns:
            df["Age"] = df["Age"]
        if "eTIV" in df.columns:
            df["eTIV"] = df["eTIV"]
        return df

    hi = _read_info(health_info_path)
    pi = _read_info(patient_info_path)

    n_edges = _n_edges(n_nodes)
    matrices = []
    groups = []
    ages = []
    sexes = []
    eTIVs = []
    subject_ids = []

    for _, row in hi.iterrows():
        sid = str(row["ID"]).strip()
        path = _find_msn_path(sid, health_msn_dir, n_nodes)
        if path is None:
            continue
        mat = load_msn_csv(path, n_nodes)
        if mat.size != n_nodes * n_nodes:
            continue
        matrices.append(matrix_to_edges(mat))
        groups.append(0)
        ages.append(float(row["Age"]))
        sexes.append(int(row["sex"]))
        eTIVs.append(float(row["eTIV"]))
        subject_ids.append(sid)

    for _, row in pi.iterrows():
        sid = str(row["ID"]).strip()
        path = _find_msn_path(sid, patient_msn_dir, n_nodes)
        if path is None:
            continue
        mat = load_msn_csv(path, n_nodes)
        if mat.size != n_nodes * n_nodes:
            continue
        matrices.append(matrix_to_edges(mat))
        groups.append(1)
        ages.append(float(row["Age"]))
        sexes.append(int(row["sex"]))
        eTIVs.append(float(row["eTIV"]))
        subject_ids.append(sid)

    edge_matrix = np.stack(matrices, axis=0).astype(np.float64)
    group = np.array(groups, dtype=np.float64)
    age = np.array(ages, dtype=np.float64)
    sex = np.array(sexes, dtype=np.float64)
    eTIV = np.array(eTIVs, dtype=np.float64)

    assert edge_matrix.shape[1] == n_edges, (edge_matrix.shape, n_edges)
    return edge_matrix, group, age, sex, eTIV, subject_ids


def load_patient_cluster_data(
    patient_msn_dir: Path,
    patient_info_path: Path,
    cluster_csv_path: Path,
    n_nodes: int = N_NODES,
    use_processed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    仅加载患者的 MSN、协变量与聚类标签。cluster_csv 需含 subject_id 与 cluster_label。

    Returns
    -------
    edge_matrix, age, sex, eTIV, cluster_label (0/1/2), subject_ids
    """
    if use_processed:
        patient_info_p = patient_info_path.parent / (patient_info_path.stem.replace("info", "info_processed") + ".csv")
        if patient_info_p.exists():
            patient_info_path = patient_info_p

    pi = pd.read_csv(patient_info_path, dtype={"ID": str})
    pi["ID"] = pi["ID"].astype(str).str.strip()
    if "Gender" in pi.columns:
        pi["sex"] = pi["Gender"].map({"f": 0, "m": 1})
    elif "Sex" in pi.columns:
        pi["sex"] = pi["Sex"]
    if "eTIV" not in pi.columns:
        pi["eTIV"] = 1.0

    cl = pd.read_csv(cluster_csv_path, dtype=str)
    id_col = "subject_id" if "subject_id" in cl.columns else "ID"
    cl["subject_id"] = cl[id_col].astype(str).str.strip()
    if "cluster_label" not in cl.columns:
        raise ValueError("cluster_csv 需包含 cluster_label 列")
    # 统一为 0,1,2
    lab = cl["cluster_label"].values.astype(np.intp)
    if lab.min() >= 1:
        lab = lab - 1
    cl = cl.assign(cluster_01=lab).set_index("subject_id")

    n_edges = _n_edges(n_nodes)
    matrices = []
    ages = []
    sexes = []
    eTIVs = []
    clusters = []
    subject_ids = []

    for _, row in pi.iterrows():
        sid = str(row["ID"]).strip()
        if sid not in cl.index:
            continue
        path = _find_msn_path(sid, patient_msn_dir, n_nodes)
        if path is None:
            continue
        mat = load_msn_csv(path, n_nodes)
        if mat.size != n_nodes * n_nodes:
            continue
        matrices.append(matrix_to_edges(mat))
        ages.append(float(row["Age"]))
        sexes.append(int(row["sex"]))
        eTIVs.append(float(row["eTIV"]))
        clusters.append(int(cl.loc[sid, "cluster_01"]))
        subject_ids.append(sid)

    clusters = np.array(clusters, dtype=np.intp)

    edge_matrix = np.stack(matrices, axis=0).astype(np.float64)
    age = np.array(ages, dtype=np.float64)
    sex = np.array(sexes, dtype=np.float64)
    eTIV = np.array(eTIVs, dtype=np.float64)
    assert edge_matrix.shape[1] == n_edges
    return edge_matrix, age, sex, eTIV, clusters, subject_ids


def build_design_nc_patient(
    group: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    eTIV: np.ndarray,
    global_strength: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    设计矩阵：edge ~ group + age + sex + eTIV（+ 可选 global_strength）。
    列顺序：[1, group, sex, age_centered, eTIV_centered] 或再加 global_strength_centered。
    对比 [0,1,0,0,0] 或 [0,1,0,0,0,0] 即 group 主效应。
    若组间“每被试边权均值”差异大，建议传入 global_strength（如 edge_matrix.mean(axis=1)）以控制全局强度。
    """
    age_c = age - np.mean(age)
    eTIV_c = eTIV - np.mean(eTIV)
    cols = [
        np.ones(len(group)),
        group,
        sex,
        age_c,
        eTIV_c,
    ]
    if global_strength is not None:
        gs_c = np.asarray(global_strength, dtype=np.float64).ravel() - np.mean(global_strength)
        cols.append(gs_c)
    X = np.column_stack(cols).astype(np.float64)
    return X


def build_design_cluster(
    cluster: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    eTIV: np.ndarray,
    reference: int = 2,
) -> np.ndarray:
    """
    多组设计：edge ~ cluster + age + sex + eTIV。
    cluster 为 0,1,2；reference 为参考组（默认 2，即第三组为参考）。
    列顺序：[1, d1, d2, sex, age_centered, eTIV_centered]，d1/d2 为前两组的哑变量。
    """
    uniq = np.unique(cluster)
    uniq = np.sort(uniq)
    if reference not in uniq:
        reference = int(uniq[-1])
    dummies = []
    for k in uniq:
        if k == reference:
            continue
        dummies.append((cluster == k).astype(np.float64))
    age_c = age - np.mean(age)
    eTIV_c = eTIV - np.mean(eTIV)
    X = np.column_stack([
        np.ones(len(cluster)),
        *dummies,
        sex,
        age_c,
        eTIV_c,
    ]).astype(np.float64)
    return X
