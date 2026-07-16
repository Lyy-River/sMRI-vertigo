from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from sklearn.metrics import adjusted_rand_score


def bootstrap_ari_stability(
    features: np.ndarray,
    cluster_fn: Callable[[np.ndarray], np.ndarray],
    n_bootstrap: int = 100,
    sample_ratio: float = 0.8,
    random_state: int = 0,
) -> Dict[str, object]:
    """
    Evaluate clustering stability with bootstrap resampling and pairwise ARI.
    """
    rng = np.random.default_rng(random_state)
    n_samples = features.shape[0]
    b_size = max(1, int(round(sample_ratio * n_samples)))

    indices_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=b_size, replace=False)
        X_b = features[idx]
        labels_b = np.asarray(cluster_fn(X_b), dtype=int)
        if labels_b.shape[0] != b_size:
            raise ValueError(
                "Length of labels returned by cluster_fn does not match bootstrap sample size."
            )
        indices_list.append(idx)
        labels_list.append(labels_b)

    ari_values: List[float] = []
    n = len(indices_list)

    for i in range(n):
        idx_i = indices_list[i]
        lab_i = labels_list[i]
        pos_map_i = {int(v): int(k) for k, v in enumerate(idx_i)}

        for j in range(i + 1, n):
            idx_j = indices_list[j]
            lab_j = labels_list[j]
            pos_map_j = {int(v): int(k) for k, v in enumerate(idx_j)}
            common = np.intersect1d(idx_i, idx_j)
            if common.size < 2:
                continue
            pos_i = np.array([pos_map_i[int(c)] for c in common], dtype=int)
            pos_j = np.array([pos_map_j[int(c)] for c in common], dtype=int)
            ari = adjusted_rand_score(lab_i[pos_i], lab_j[pos_j])
            ari_values.append(float(ari))

    if ari_values:
        ari_arr = np.asarray(ari_values, dtype=float)
        mean_ari = float(ari_arr.mean())
        std_ari = float(ari_arr.std(ddof=1)) if ari_arr.size > 1 else 0.0
        ci_low, ci_high = np.percentile(ari_arr, [2.5, 97.5])
    else:
        ari_arr = np.array([], dtype=float)
        mean_ari = std_ari = ci_low = ci_high = float("nan")

    summary = {
        "mean_ari": mean_ari,
        "std_ari": std_ari,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }
    return {
        "indices": indices_list,
        "labels": labels_list,
        "ari_pairwise": ari_arr,
        "summary": summary,
    }
