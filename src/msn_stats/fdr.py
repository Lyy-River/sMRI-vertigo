"""FDR 校正（Benjamini–Hochberg）。"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fdr_correct(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR 校正。

    Returns
    -------
    rejected : (n,) bool，是否拒绝零假设
    p_fdr : (n,) 经 FDR 校正后的 q 值（若 q < alpha 则拒绝）
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    # BH: q_(i) = (n/i)*p_(i)，再保证单调性 Q_(i) = min(q_(i), q_(i+1), ..., q_(n))，即从末尾向前取累积最小
    val = n / np.arange(1, n + 1, dtype=np.float64) * p_sorted
    cummin_from_end = np.minimum.accumulate(val[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(1.0, cummin_from_end)
    rejected = q < alpha
    return rejected, q
