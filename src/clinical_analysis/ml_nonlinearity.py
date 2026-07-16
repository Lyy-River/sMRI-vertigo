# -*- coding: utf-8 -*-
"""
ML-based feature selection and linear vs nonlinear comparison.
Logistic regression vs Random Forest, permutation importance.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from . import config as _config


def _prepare_Xy(
    df: pd.DataFrame,
    feature_list: List[str],
    label_col: str = "Cluster",
    fill_missing: str = "median",
    min_samples: int = 50,
) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    """Prepare X, y from merged dataset; return (X, y, feature_names) or None."""
    available = [f for f in feature_list if f in df.columns]
    if len(available) < len(_config.COVARIATES):
        return None
    X = df[available].copy()
    y = df[label_col].copy()
    # Object -> numeric
    for col in X.columns:
        if X[col].dtype == object:
            s = X[col].astype(str).str.strip()
            s = s.str.replace(r"^<(\d*\.?\d+)$", r"\1", regex=True)
            s = s.str.replace(r"^>(\d*\.?\d+)$", r"\1", regex=True)
            X[col] = pd.to_numeric(s, errors="coerce")
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    if len(X) < min_samples:
        return None
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    X = X.dropna(axis=1)
    available = [f for f in available if f in X.columns]
    if len(available) < len(_config.COVARIATES):
        return None
    X = X[available]
    return (X, y, available)


def run_rf_feature_selection(
    datasets: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    cv_folds: int = 5,
    random_state: int = 42,
    n_repeats: int = 20,
) -> Dict[str, Any]:
    """
    For each data category (biochem, cbc, ecg, diagnosis), fit RF and return permutation importance.
    Also compare LR vs RF CV accuracy; return summary and importance tables.

    Returns dict with:
      - summary: list of dicts (category, lr_acc, rf_acc, accuracy_diff, p_value, supports_nonlinear)
      - importance: dict category -> DataFrame (feature, importance_mean, importance_std)
    """
    if config is None:
        config = datasets.get("config", _config.DEFAULT_CONFIG)
    label_col = config.get("cluster_label_col", _config.CLUSTER_LABEL_COL)
    covars = config.get("covariates", _config.COVARIATES)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    summary: List[Dict[str, Any]] = []
    importance: Dict[str, pd.DataFrame] = {}

    categories = [
        ("biochem", datasets.get("biochem"), config.get("biochem_features", [])),
        ("cbc", datasets.get("cbc"), config.get("blood_features", [])),
        ("ecg", datasets.get("ecg"), config.get("ecg_features", [])),
        ("diagnosis", datasets.get("diagnosis"), config.get("diagnosis_features", [])),
    ]

    for name, df, feat_list in categories:
        if df is None or not feat_list:
            continue
        feature_list = covars + [f for f in feat_list if f != "Cluster" and f in df.columns]
        out = _prepare_Xy(df, feature_list, label_col=label_col)
        if out is None:
            continue
        X, y, available = out
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        X_scaled = X_scaled.astype(float)
        y = y.astype(int)

        lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=random_state)
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_split=20, min_samples_leaf=10,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
        lr_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring="accuracy")
        rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
        diff = rf_scores.mean() - lr_scores.mean()
        try:
            _, p_val = ttest_rel(rf_scores, lr_scores)
        except Exception:
            p_val = 1.0
        summary.append({
            "category": name,
            "n": len(X),
            "lr_cv_accuracy_mean": lr_scores.mean(),
            "lr_cv_accuracy_std": lr_scores.std(),
            "rf_cv_accuracy_mean": rf_scores.mean(),
            "rf_cv_accuracy_std": rf_scores.std(),
            "accuracy_diff": diff,
            "p_value": p_val,
            "supports_nonlinear": (diff > 0) and (p_val < 0.05),
        })

        rf.fit(X_scaled, y)
        perm = permutation_importance(rf, X_scaled, y, n_repeats=n_repeats, scoring="accuracy", random_state=random_state, n_jobs=-1)
        imp_df = pd.DataFrame({
            "feature": X_scaled.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False)
        importance[name] = imp_df

    return {"summary": summary, "importance": importance}
