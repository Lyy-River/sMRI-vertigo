# -*- coding: utf-8 -*-
"""
Plotting utilities: boxplots, KDE plots, forest plots.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def plot_boxplots_by_cluster(
    df: pd.DataFrame,
    variables: List[str],
    cluster_col: str = "Cluster",
    output_path: Optional[Path] = None,
    figsize: tuple = (6, 4),
) -> Any:
    """Draw boxplots of variables by cluster. Save if output_path set."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    n = len(variables)
    if n == 0:
        return None
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, var in enumerate(variables):
        if i >= len(axes) or var not in df.columns:
            continue
        ax = axes[i]
        df.boxplot(column=var, by=cluster_col, ax=ax)
        ax.set_title(var)
        ax.set_xlabel(cluster_col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("")  # remove default
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_forest_or(
    results_df: pd.DataFrame,
    odds_ratio_col: str = "OR",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    label_col: str = "variable",
    output_path: Optional[Path] = None,
) -> Any:
    """Forest plot for OR (and CI). results_df must have OR, ci_low, ci_high, label."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if odds_ratio_col not in results_df.columns or label_col not in results_df.columns:
        return None
    df = results_df.copy()
    df = df.dropna(subset=[odds_ratio_col, label_col])
    if ci_low_col in df.columns and ci_high_col in df.columns:
        df["ci_low"] = df[ci_low_col]
        df["ci_high"] = df[ci_high_col]
    else:
        df["ci_low"] = df[odds_ratio_col]
        df["ci_high"] = df[odds_ratio_col]
    n = len(df)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.3)))
    y_pos = range(n)
    ax.errorbar(
        df[odds_ratio_col],
        y_pos,
        xerr=[df[odds_ratio_col] - df["ci_low"], df["ci_high"] - df[odds_ratio_col]],
        fmt="o",
        capsize=4,
    )
    ax.axvline(1, color="gray", linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df[label_col].tolist())
    ax.set_xlabel("Odds Ratio")
    ax.set_title("Forest plot (OR with 95% CI)")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
