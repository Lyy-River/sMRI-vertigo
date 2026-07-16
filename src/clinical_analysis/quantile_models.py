# -*- coding: utf-8 -*-
"""
Quantile regression: model effect of Cluster (and covariates) on quantiles of continuous outcomes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
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


def run_quantile_models(
    datasets: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    quantiles: Optional[List[float]] = None,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Fit quantile regression: Y ~ C(Cluster) + age + sex (+ optional time_diff)
    for quantiles 0.25, 0.50, 0.75 (or config quantiles).

    Returns DataFrame with columns: data, variable, quantile, cluster_effect_pvalue, p_fdr, (and optional coef summary).
    For simplicity we report the overall Cluster effect (e.g. F-test or joint test if available).
    statsmodels QuantReg does not provide a single "cluster effect" F-test; we report the min p-value across cluster dummies per quantile.
    """
    if config is None:
        config = datasets.get("config", _config.DEFAULT_CONFIG)
    label_col = config.get("cluster_label_col", _config.CLUSTER_LABEL_COL)
    covars = config.get("covariates", _config.COVARIATES)
    time_col = config.get("time_diff_col")
    qs = quantiles if quantiles is not None else config.get("quantiles", _config.QUANTILES)

    rows: List[Dict[str, Any]] = []

    for name, df, features in _get_continuous_datasets(datasets, config):
        use = [label_col] + covars + ([] if not time_col or time_col not in df.columns else [time_col])
        formula_cov = " + ".join([f"C({label_col})"] + covars + ([time_col] if time_col and time_col in df.columns else []))
        df_clean = df[use + features].copy()
        df_clean = df_clean.dropna()
        df_clean[label_col] = df_clean[label_col].astype(int)

        for var in features:
            if var not in df_clean.columns or not np.issubdtype(df_clean[var].dtype, np.number):
                continue
            y = df_clean[var]
            X = df_clean[[label_col] + covars]
            if time_col and time_col in df_clean.columns:
                X = X.join(df_clean[[time_col]])
            # Build X: constant + Cluster dummies (drop_first for reference) + covariates
            dummies = pd.get_dummies(df_clean[[label_col]], prefix="Cluster", drop_first=True)
            X = pd.DataFrame(sm.add_constant(dummies, has_constant="add"), index=df_clean.index)
            for c in covars:
                if c in df_clean.columns:
                    X[c] = df_clean[c].values
            if time_col and time_col in df_clean.columns:
                X[time_col] = df_clean[time_col].values
            X = X.dropna(axis=1)
            valid = y.notna() & X.notna().all(axis=1)
            y = y[valid]
            X = X.loc[valid].astype(float)
            if len(y) < 20:
                continue

            for q in qs:
                try:
                    model = QuantReg(y, X).fit(q=q)
                    cluster_params = [p for p in model.params.index if p.startswith("Cluster")]
                    if not cluster_params:
                        continue
                    pvals = model.pvalues
                    cluster_pvals = [float(pvals.loc[p]) for p in cluster_params if p in pvals.index]
                    cluster_effect_pvalue = min(cluster_pvals) if cluster_pvals else np.nan
                    rows.append({
                        "data": name,
                        "variable": var,
                        "quantile": q,
                        "cluster_effect_pvalue": cluster_effect_pvalue,
                        "n": len(y),
                    })
                except Exception:
                    continue

    if not rows:
        return pd.DataFrame(columns=["data", "variable", "quantile", "cluster_effect_pvalue", "p_fdr", "n"])

    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["cluster_effect_pvalue"], method="fdr_bh")[1]
    return res
