"""
逐边 GLM（两组）与 ANCOVA（多组）。

- 两组：edge ~ group + age + sex + eTIV，输出 t、p、beta_group
- 多组：edge ~ cluster_dummies + age + sex + eTIV，输出 F、p；可选返回各对比的 t/p 供 post-hoc
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _ols_fit(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Y = X @ B + E。返回 B (p, E), R (n, E), MSE (E,), iXtX (p, p)。"""
    n, p = X.shape
    E = Y.shape[1]
    XtX = X.T @ X
    iXtX = np.linalg.inv(XtX)
    B = iXtX @ (X.T @ Y)
    R = Y - X @ B
    df = n - p
    MSE = np.sum(R * R, axis=0) / np.maximum(df, 1)
    return B, R, MSE, iXtX


def edgewise_glm_twogroup(
    edge_matrix: np.ndarray,
    design: np.ndarray,
    group_col: int = 1,
    mse_floor_percentile: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    逐边 GLM，检验 design 中第 group_col 列（group）的效应。
    design 列顺序约定：[1, group, sex, age_centered, eTIV_centered]，group_col=1。

    mse_floor_percentile : 将每条边 MSE 压下到“全体 MSE 的该分位数”以上，避免残差过小导致 t 爆炸；
        常用 5~10；若仍出现几乎全显著，可试 15~20 或检查是否需控制 global strength。

    Returns
    -------
    t : (n_edges,) t 统计量
    p : (n_edges,) 双尾 p 值
    beta : (n_edges,) 对应列的回归系数
    """
    n, p = design.shape
    n_edges = edge_matrix.shape[1]
    contrast = np.zeros(p, dtype=np.float64)
    contrast[group_col] = 1.0

    B, R, MSE, iXtX = _ols_fit(design, edge_matrix)
    c = contrast
    effect = c @ B
    c_iXtX_c = float(c @ iXtX @ c)
    # 用 MSE 的 mse_floor_percentile 分位数作为下界，避免 t 爆炸、p 全为 0
    mse_floor = np.maximum(
        np.percentile(MSE[MSE > 0], mse_floor_percentile) if np.any(MSE > 0) else 1e-10,
        1e-10,
    )
    mse_safe = np.maximum(MSE, mse_floor)
    var_effect = c_iXtX_c * mse_safe
    se = np.sqrt(np.maximum(var_effect, 1e-12))
    t = effect / se
    df = n - p
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(np.abs(t), df))
    p = np.clip(p, 0, 1)
    return t, p, effect


def edgewise_ancova_multigroup(
    edge_matrix: np.ndarray,
    design: np.ndarray,
    n_group_cols: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    逐边 ANCOVA：多组（cluster 哑变量）+ 协变量。
    design 前 n_group_cols 列为组哑变量（不含参考组），检验“至少一组不同”即联合检验这 n_group_cols 个系数为 0。

    Returns
    -------
    F : (n_edges,) F 统计量
    p : (n_edges,) p 值
    """
    n, p = design.shape
    df1 = n_group_cols
    df2 = n - p

    B, R, MSE, iXtX = _ols_fit(design, edge_matrix)
    # 简化：仅检验前 n_group_cols 个系数全为 0 的 F 检验
    # F = (R0^2 - R1^2)/df1 / (R1^2/df2)，其中 R1^2 = SSE = sum(R^2)。这里用 Wald: (C@B)' (C @ (X'X)^{-1} C')^{-1} (C@B) / (df1 * MSE)
    C = np.eye(p, p, dtype=np.float64)[:n_group_cols]
    CB = C @ B
    # Cov(CB) = C (X'X)^{-1} C' * MSE (每边一个标量 MSE_e)
    CiXtXCt = C @ iXtX @ C.T
    try:
        iCiXtXCt = np.linalg.inv(CiXtXCt)
    except np.linalg.LinAlgError:
        F = np.zeros(edge_matrix.shape[1])
        pval = np.ones(edge_matrix.shape[1])
        return F, pval
    # 对每条边：F = (CB)' @ iCiXtXCt @ (CB) / df1 / MSE；MSE 过小时设下界避免 F 爆炸
    mse_floor = np.maximum(np.percentile(MSE[MSE > 0], 10.0) if np.any(MSE > 0) else 1e-10, 1e-10)
    mse_safe = np.maximum(MSE, mse_floor)
    F_per_edge = np.sum(CB * (iCiXtXCt @ CB), axis=0) / np.maximum(df1 * mse_safe, 1e-12)
    from scipy import stats
    pval = 1 - stats.f.cdf(F_per_edge, df1, df2)
    pval = np.clip(pval, 0, 1)
    return F_per_edge, pval


def edgewise_glm_contrast(
    edge_matrix: np.ndarray,
    design: np.ndarray,
    contrast: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    逐边 GLM 单对比：c'B，返回 t 与 p（双尾）。用于 post-hoc 两两比较。
    """
    B, R, MSE, iXtX = _ols_fit(design, edge_matrix)
    c = np.asarray(contrast, dtype=np.float64).ravel()
    effect = c @ B
    c_iXtX_c = float(c @ iXtX @ c)
    mse_floor = np.maximum(
        np.percentile(MSE[MSE > 0], 10.0) if np.any(MSE > 0) else 1e-10,
        1e-10,
    )
    mse_safe = np.maximum(MSE, mse_floor)
    se = np.sqrt(np.maximum(c_iXtX_c * mse_safe, 1e-12))
    t = effect / se
    df = design.shape[0] - design.shape[1]
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(np.abs(t), df))
    return t, np.clip(p, 0, 1)
