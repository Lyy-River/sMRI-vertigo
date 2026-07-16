# -*- coding: utf-8 -*-
"""
Logistic regression for binary outcomes (ECG, diagnosis indicators).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from . import config as _config


def run_logistic_models(
    datasets: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each binary outcome in ECG and diagnosis data, fit:
    outcome ~ C(Cluster) + age + sex (+ optional time_diff).
    Report LRT p-value for Cluster effect, with FDR correction.

    Returns DataFrame: data, variable, p_value, p_fdr, significant.
    """
    config = datasets.get("config") or config or _config.DEFAULT_CONFIG
    if config is None:
        config = _config.DEFAULT_CONFIG
    label_col = config.get("cluster_label_col", _config.CLUSTER_LABEL_COL)
    covars = config.get("covariates", _config.COVARIATES)
    time_col = config.get("time_diff_col")
    rows: List[Dict[str, Any]] = []

    for name, key, feat_list in [
        ("ecg", "ecg", config.get("ecg_features") or []),
        ("diagnosis", "diagnosis", config.get("diagnosis_features") or []),
    ]:
        df = datasets.get(key)
        if df is None:
            continue
        feat_list = [f for f in feat_list if f in df.columns]
        if not feat_list and key == "diagnosis":
            skip = {label_col, config.get("cluster_id_col", "SubjectID")} | set(covars)
            if time_col and time_col in df.columns:
                skip.add(time_col)
            feat_list = [
                c
                for c in df.columns
                if c not in skip
                and (np.issubdtype(df[c].dtype, np.number) or df[c].dtype == bool)
            ]
        if not feat_list:
            continue
        formula_rhs = f"C({label_col}) + " + " + ".join(covars)
        if time_col and time_col in df.columns:
            formula_rhs += f" + {time_col}"
        for var in feat_list:
            if var not in df.columns:
                continue
            use_cols = [var, label_col] + covars + (
                [time_col] if time_col and time_col in df.columns else []
            )
            data = df[use_cols].dropna()
            if data[var].nunique() < 2 or len(data) < 20:
                continue
            data_fit = data[use_cols].copy()
            data_fit = data_fit.rename(columns={var: "y"})
            data_fit["y"] = pd.to_numeric(data_fit["y"], errors="coerce").fillna(0).astype(int)
            if data_fit["y"].nunique() < 2 or len(data_fit) < 20:
                continue
            try:
                model = smf.logit("y ~ " + formula_rhs, data=data_fit).fit(disp=0)
                restricted_rhs = " + ".join(covars) + (
                    f" + {time_col}" if time_col and time_col in data_fit.columns else ""
                )
                restricted = smf.logit("y ~ " + restricted_rhs, data=data_fit).fit(disp=0)
                lr_stat = 2 * (model.llf - restricted.llf)
                from scipy import stats as scipy_stats

                df_diff = model.df_model - restricted.df_model
                p_val = 1 - scipy_stats.chi2.cdf(lr_stat, df_diff) if df_diff > 0 else 1.0
                rows.append({"data": name, "variable": var, "p_value": p_val})
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(columns=["data", "variable", "p_value", "p_fdr", "significant"])

    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["significant"] = res["p_fdr"] < fdr_alpha
    return res
