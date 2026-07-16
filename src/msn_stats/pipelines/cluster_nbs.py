from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from msn_stats.data import (
    N_NODES,
    build_design_cluster,
    load_patient_cluster_data,
)
from msn_stats.edgewise import (
    edgewise_ancova_multigroup,
)
from msn_stats.nbs_glm import (
    NBSResult,
    run_between_group_stats,
)
from msn_stats.posthoc import (
    build_cluster_contrasts,
    pairwise_ttest_fdr,
)


def run_patient_cluster_nbs(
    patient_msn_dir: Path,
    patient_info_path: Path,
    cluster_csv_path: Path,
    results_root: Path,
    primary_threshold: float = 6.0,
    n_perm: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """
    Run ANCOVA + NBS + post-hoc tests for patient clusters.
    """
    edge_matrix, age, sex, eTIV, clusters, subject_ids = load_patient_cluster_data(
        patient_msn_dir=patient_msn_dir,
        patient_info_path=patient_info_path,
        cluster_csv_path=cluster_csv_path,
    )

    design = build_design_cluster(
        cluster=clusters,
        age=age,
        sex=sex,
        eTIV=eTIV,
        reference=2,
    )

    # ANCOVA F-test for overall cluster effect (per-edge)
    F_vals, p_vals = edgewise_ancova_multigroup(
        edge_matrix=edge_matrix,
        design=design,
        n_group_cols=2,
    )

    # NBS on same design: components where any cluster differs
    matrices = edge_matrix.reshape(-1, N_NODES, N_NODES)
    # 只针对第一个组哑变量的对比，可以按需扩展
    contrast = np.zeros(design.shape[1], dtype=np.float64)
    contrast[1] = 1.0

    nbs_res: NBSResult = run_between_group_stats(
        msn_matrices=matrices,
        design_matrix=design,
        contrast=contrast,
        n_nodes=N_NODES,
        primary_threshold=primary_threshold,
        n_perm=n_perm,
        two_tailed=True,
        seed=seed,
    )

    # Post-hoc pairwise tests with FDR per contrast
    contrasts = build_cluster_contrasts(pred_dim=design.shape[1], reference=2)
    posthoc_results = pairwise_ttest_fdr(
        edge_matrix=edge_matrix,
        design=design,
        contrast_pairs=contrasts,
        alpha=0.05,
    )

    out_dir = results_root / "cluster_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存 ANCOVA F & p
    ancova_csv = out_dir / "cluster_ancova_edgewise.csv"
    pd.DataFrame({"F": F_vals, "p": p_vals}).to_csv(ancova_csv, index=False)

    # 简单保存 post-hoc 拒绝结果
    for label, (_t, _p, rejected) in posthoc_results.items():
        out = pd.DataFrame({"rejected": rejected.astype(int)})
        out.to_csv(out_dir / f"posthoc_{label}.csv", index=False)

    return {
        "edge_matrix": edge_matrix,
        "age": age,
        "sex": sex,
        "eTIV": eTIV,
        "clusters": clusters,
        "subject_ids": subject_ids,
        "design": design,
        "ancova_F": F_vals,
        "ancova_p": p_vals,
        "nbs_result": nbs_res,
        "posthoc_results": posthoc_results,
        "ancova_csv": ancova_csv,
    }

