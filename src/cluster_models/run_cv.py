# -*- coding: utf-8 -*-
"""CV 运行与结果汇总：单模型/多模型 CV、排列重要性。"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance

from .models import get_model


def run_single_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: tuple = ("accuracy", "neg_log_loss"),
    random_state: int = 42,
) -> Dict[str, float]:
    """
    对单个模型跑 K 折 CV，返回统一格式的指标。

    Returns:
        {"accuracy_mean", "accuracy_std", "log_loss_mean", "log_loss_std"}
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    out: Dict[str, float] = {}

    if "accuracy" in scoring:
        acc_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy", n_jobs=-1)
        out["accuracy_mean"] = float(acc_scores.mean())
        out["accuracy_std"] = float(acc_scores.std())

    if "neg_log_loss" in scoring:
        nll_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_log_loss", n_jobs=-1)
        out["log_loss_mean"] = float(-nll_scores.mean())
        out["log_loss_std"] = float(nll_scores.std())

    return out


def run_models_cv(
    model_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    model_params: Optional[Dict[str, dict]] = None,
    n_classes: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    对多个模型名批量跑 CV，返回汇总 DataFrame。

    model_params: 如 {"xgb": {"max_depth": 4}, "rf": {"max_depth": 6}}
    """
    model_params = model_params or {}
    rows = []
    for name in model_names:
        try:
            clf = get_model(name, n_classes=n_classes, **model_params.get(name, {}))
        except ImportError as e:
            if "xgb" in name.lower():
                rows.append(
                    {
                        "model_name": name,
                        "accuracy_mean": np.nan,
                        "accuracy_std": np.nan,
                        "log_loss_mean": np.nan,
                        "log_loss_std": np.nan,
                        "error": str(e),
                    }
                )
                continue
            raise
        res = run_single_cv(clf, X, y, cv=cv, random_state=random_state)
        rows.append(
            {
                "model_name": name,
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "log_loss_mean": res["log_loss_mean"],
                "log_loss_std": res["log_loss_std"],
            }
        )
    return pd.DataFrame(rows)


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 20,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    计算排列重要性，返回 DataFrame(feature, importance_mean, importance_std)。
    """
    model.fit(X, y)
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return df.reset_index(drop=True)
