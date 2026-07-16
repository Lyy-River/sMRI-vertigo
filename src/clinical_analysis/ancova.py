# -*- coding: utf-8 -*-
"""
ANCOVA / OLS models for continuous clinical variables: mean effect testing.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.stats.multitest import multipletests

from . import config as _config


def _get_continuous_datasets(datasets: Dict[str, Any], config: Dict[str, Any]) -> List[tuple]:
    out: List[tuple] = []
    if datasets.get("biochem") is not None:
        feats = config.get("biochem_features", [])
        if feats:
            out.append(("biochem", datasets["biochem"], feats))
    if datasets.get("cbc") is not None:
        feats = config.get("blood_features", [])
        if feats:
            out.append(("cbc", datasets["cbc"], feats))
    return out


def run_ancova_models(
    datasets: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    ANCOVA: Y ~ C(Cluster) + age + sex (+ optional abs_time_diff_days).
    One model per continuous variable; report F and p for C(Cluster), then FDR correction.

    Returns DataFrame: data, variable, F_value, p_value, p_fdr, significant.
    """
    if config is None:
        config = datasets.get("config", _config.DEFAULT_CONFIG)
    label_col = config.get("cluster_label_col", _config.CLUSTER_LABEL_COL)
    covars = config.get("covariates", _config.COVARIATES)
    time_col = config.get("time_diff_col")

    rows: List[Dict[str, Any]] = []

    for name, df, features in _get_continuous_datasets(datasets, config):
        formula_rhs = f"C({label_col}) + " + " + ".join(covars)
        if time_col and time_col in df.columns:
            formula_rhs += f" + {time_col}"
        df_work = df.dropna(subset=[label_col] + covars + ([] if not time_col else [time_col]))

        for var in features:
            if var not in df_work.columns or not np.issubdtype(df_work[var].dtype, np.number):
                continue
            use_cols = [var, label_col] + covars + ([time_col] if time_col and time_col in df_work.columns else [])
            data = df_work[use_cols].dropna()
            if len(data) < 20:
                continue
            formula = f"{var} ~ " + formula_rhs
            try:
                model = smf.ols(formula, data=data).fit()
                anova = sms.anova_lm(model, typ=2)
                if f"C({label_col})" not in anova.index:
                    continue
                f_val = float(anova.loc[f"C({label_col})", "F"])
                p_val = float(anova.loc[f"C({label_col})", "PR(>F)"])
                rows.append({"data": name, "variable": var, "F_value": f_val, "p_value": p_val})
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(columns=["data", "variable", "F_value", "p_value", "p_fdr", "significant"])

    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["significant"] = res["p_fdr"] < fdr_alpha
    return res
