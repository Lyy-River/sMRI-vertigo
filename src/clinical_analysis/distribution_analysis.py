# -*- coding: utf-8 -*-
"""
Distribution-level analysis: KDE visualization and KS tests for equality of distributions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from . import config as _config


def _get_continuous_datasets(datasets: Dict[str, Any], config: Dict[str, Any]) -> List[Tuple[str, pd.DataFrame, List[str]]]:
    """Return list of (name, df, feature_columns) for continuous outcomes (biochem, cbc)."""
    out: List[Tuple[str, pd.DataFrame, List[str]]] = []
    if datasets.get("biochem") is not None:
        feats = config.get("biochem_features", [])
        if feats:
            out.append(("biochem", datasets["biochem"], feats))
    if datasets.get("cbc") is not None:
        feats = config.get("blood_features", [])
        if feats:
            out.append(("cbc", datasets["cbc"], feats))
    return out


def run_ks_distribution_tests(
    datasets: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Two-sample Kolmogorov-Smirnov test for each pair of clusters and each continuous variable.

    Returns DataFrame with columns: variable, pair, ks_stat, p_value, p_fdr.
    """
    if config is None:
        config = datasets.get("config", _config.DEFAULT_CONFIG)
    label_col = config.get("cluster_label_col", _config.CLUSTER_LABEL_COL)
    rows: List[Dict[str, Any]] = []

    for name, df, features in _get_continuous_datasets(datasets, config):
        df = df.dropna(subset=[label_col] + features)
        clusters = sorted(df[label_col].unique().astype(int))
        if len(clusters) < 2:
            continue
        for var in features:
            if var not in df.columns or not np.issubdtype(df[var].dtype, np.number):
                continue
            series = df[var].dropna()
            if len(series) < 10:
                continue
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    c1, c2 = clusters[i], clusters[j]
                    s1 = df.loc[df[label_col] == c1, var].dropna()
                    s2 = df.loc[df[label_col] == c2, var].dropna()
                    if len(s1) < 5 or len(s2) < 5:
                        continue
                    stat, p = stats.ks_2samp(s1, s2)
                    rows.append({
                        "data": name,
                        "variable": var,
                        "pair": f"Cluster{c1}_vs_Cluster{c2}",
                        "ks_stat": stat,
                        "p_value": p,
                    })

    if not rows:
        return pd.DataFrame(columns=["data", "variable", "pair", "ks_stat", "p_value", "p_fdr"])

    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["p_value"], method="fdr_bh")[1]
    return res


def run_kde_visualization(
    datasets: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    top_n_per_group: int = 6,
) -> Optional[Dict[str, Any]]:
    """
    Plot KDE curves per cluster for continuous variables (exploratory).
    Saves one figure per variable (or per first top_n_per_group variables) if output_dir is set.

    Returns dict of { variable_name: (fig, ax) } for further use if not saving.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if config is None:
        config = datasets.get("config", _config.DEFAULT_CONFIG)
    label_col = config.get("cluster_label_col", _config.CLUSTER_LABEL_COL)
    figures: Dict[str, Any] = {}

    for name, df, features in _get_continuous_datasets(datasets, config):
        df = df.dropna(subset=[label_col])
        clusters = sorted(df[label_col].unique().astype(int))
        vars_to_plot = features[:top_n_per_group] if top_n_per_group else features
        for var in vars_to_plot:
            if var not in df.columns or not np.issubdtype(df[var].dtype, np.number):
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            for c in clusters:
                ser = df.loc[df[label_col] == c, var].dropna()
                if len(ser) < 3:
                    continue
                ser.plot.kde(ax=ax, label=f"Cluster {c} (n={len(ser)})")
            ax.set_title(f"KDE: {var} ({name})")
            ax.set_xlabel(var)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            key = f"{name}_{var}"
            figures[key] = (fig, ax)
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_dir / f"kde_{key}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
    return figures if figures else None
