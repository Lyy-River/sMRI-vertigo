"""
事后两两比较：对每条边做 cluster i vs j 的 t 检验，再对全部边做 FDR 校正。

可选：Tukey HSD 需对每条边做一次，实现为对 10878 条边分别做 one-way ANOVA + Tukey；
这里采用“两两 t 检验 + 全边 FDR”，更常用且与 prompt 一致。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .edgewise import edgewise_glm_contrast
from .fdr import fdr_correct


def pairwise_ttest_fdr(
    edge_matrix: np.ndarray,
    design: np.ndarray,
    contrast_pairs: list,
    alpha: float = 0.05,
) -> Dict:
    """
    对多组设计，做两两对比的逐边 t 检验，再对“该对比下所有边”做 FDR。

    Parameters
    ----------
    edge_matrix : (n_subj, n_edges)
    design : (n_subj, n_pred)
    contrast_pairs : list of (label, contrast_vector)，contrast_vector 长度为 n_pred
    alpha : FDR 水平

    Returns
    -------
    results : dict, key=label，value=(t, p, rejected)
    """
    results = {}
    for label, contrast in contrast_pairs:
        t, p = edgewise_glm_contrast(edge_matrix, design, contrast)
        rejected, _q = fdr_correct(p, alpha=alpha)
        results[label] = (t, p, rejected)
    return results


def build_cluster_contrasts(pred_dim: int, reference: int = 2) -> list:
    """
    三组时 design 列 [1, d1, d2, sex, age, eTIV]，参考组为 cluster 2（第三组）。
    返回 [(label, contrast), ...]，contrast 长度为 pred_dim。
    """
    out = []
    if pred_dim >= 6:
        # cluster0 vs cluster2: d1=1
        c = np.zeros(pred_dim, dtype=np.float64)
        c[1] = 1.0
        out.append(("C0_vs_C2", c))
        # cluster1 vs cluster2: d2=1
        c = np.zeros(pred_dim, dtype=np.float64)
        c[2] = 1.0
        out.append(("C1_vs_C2", c))
        # cluster0 vs cluster1: d1 - d2
        c = np.zeros(pred_dim, dtype=np.float64)
        c[1], c[2] = 1.0, -1.0
        out.append(("C0_vs_C1", c))
    return out
