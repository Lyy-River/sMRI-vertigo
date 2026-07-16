from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from msn_stats.data import (
    N_NODES,
    build_design_nc_patient,
    load_nc_patient_data,
)
from msn_stats.nbs_glm import (
    NBSResult,
    component_mask_to_matrix,
    run_between_group_stats,
)


def run_nc_vs_patient_nbs(
    health_msn_dir: Path,
    patient_msn_dir: Path,
    health_info_path: Path,
    patient_info_path: Path,
    results_root: Path,
    primary_threshold: float = 6.0,
    n_perm: int = 5000,
    two_tailed: bool = True,
    seed: Optional[int] = None,
    use_global_strength: bool = True,
) -> Dict[str, object]:
    """
    End-to-end NC vs patient NBS analysis.

    Returns
    -------
    dict with keys:
        edge_matrix, group, age, sex, eTIV, subject_ids,
        design, contrast, nbs_result, summary_csv, npy_path
    """
    edge_matrix, group, age, sex, eTIV, subject_ids = load_nc_patient_data(
        health_msn_dir=health_msn_dir,
        patient_msn_dir=patient_msn_dir,
        health_info_path=health_info_path,
        patient_info_path=patient_info_path,
    )

    # Optional global strength covariate
    global_strength = None
    if use_global_strength and edge_matrix.size > 0:
        global_strength = edge_matrix.mean(axis=1)

    design = build_design_nc_patient(
        group=group,
        age=age,
        sex=sex,
        eTIV=eTIV,
        global_strength=global_strength,
    )
    # group 主效应对比：第二列为 group
    contrast = np.zeros(design.shape[1], dtype=np.float64)
    contrast[1] = 1.0

    matrices_all = edge_matrix.reshape(-1, N_NODES, N_NODES)

    nbs_res: NBSResult = run_between_group_stats(
        msn_matrices=matrices_all,
        design_matrix=design,
        contrast=contrast,
        n_nodes=N_NODES,
        primary_threshold=primary_threshold,
        n_perm=n_perm,
        two_tailed=two_tailed,
        seed=seed,
    )

    nbs_dir = results_root / "nbs"
    nbs_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"nbs_nc_vs_patient_primary{int(primary_threshold)}_perm{n_perm}"
    npy_path = nbs_dir / f"{out_name}.npy"
    np.save(
        npy_path,
        {
            "t_map": nbs_res.t_map,
            "component_masks": nbs_res.component_masks,
            "component_pvalues": nbs_res.component_pvalues,
            "component_sizes": nbs_res.component_sizes,
            "threshold": nbs_res.threshold,
            "n_perm": nbs_res.n_perm,
            "n_nodes": nbs_res.n_nodes,
        },
        allow_pickle=True,
    )

    summary = pd.DataFrame(
        {
            "component": np.arange(nbs_res.n_components),
            "n_edges": nbs_res.component_sizes,
            "p_fwer": nbs_res.component_pvalues,
        }
    )
    summary_csv = nbs_dir / f"{out_name}.csv"
    summary.to_csv(summary_csv, index=False)

    for k, mask in enumerate(nbs_res.component_masks):
        mat = component_mask_to_matrix(mask, nbs_res.n_nodes)
        np.save(nbs_dir / f"{out_name}_component{k}.npy", mat)

    return {
        "edge_matrix": edge_matrix,
        "group": group,
        "age": age,
        "sex": sex,
        "eTIV": eTIV,
        "subject_ids": subject_ids,
        "design": design,
        "contrast": contrast,
        "nbs_result": nbs_res,
        "summary_csv": summary_csv,
        "npy_path": npy_path,
    }

