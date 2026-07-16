from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from msn_stats.fdr import fdr_correct


def run_topology_glm(
    health_topology_csv: Path,
    patient_topology_csv: Path,
    results_root: Path,
) -> Dict[str, object]:
    """
    Simple GLM / t-test on global topology metrics between health and patient.

    Currently implements independent t-tests for each metric with FDR correction.
    """
    h_df = pd.read_csv(health_topology_csv)
    p_df = pd.read_csv(patient_topology_csv)

    # assume columns: subject_id, group, metric1, metric2, ...
    metric_cols = [
        c for c in h_df.columns if c not in ("subject_id", "group")
    ]

    rows = []
    p_values = []
    for col in metric_cols:
        h_vals = h_df[col].to_numpy(dtype=float)
        p_vals = p_df[col].to_numpy(dtype=float)
        t_stat, p_val = stats.ttest_ind(h_vals, p_vals, equal_var=False, nan_policy="omit")
        rows.append({"metric": col, "t": t_stat, "p_uncorrected": p_val})
        p_values.append(p_val)

    p_values_arr = np.array(p_values, dtype=float)
    _, q_vals = fdr_correct(p_values_arr, alpha=0.05)

    for row, q in zip(rows, q_vals):
        row["q_fdr"] = q
        row["significant"] = bool(q < 0.05)

    out_df = pd.DataFrame(rows)
    out_dir = results_root / "topology_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "health_vs_patient_topology_glm.csv"
    out_df.to_csv(out_csv, index=False)

    return {
        "results": out_df,
        "output_csv": out_csv,
    }

