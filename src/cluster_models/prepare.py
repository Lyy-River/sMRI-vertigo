# -*- coding: utf-8 -*-
"""数据准备：从 DataFrame 提取 X/y，清洗、标准化，供 CV 使用。"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_Xy(
    df: pd.DataFrame,
    feature_list: List[str],
    target: str = "Cluster",
    covariates: Optional[List[str]] = None,
    fill_missing: str = "median",
    scale: bool = True,
    min_samples: int = 50,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    从已含 Cluster 的 DataFrame 中按特征列表取出 X、y，做缺失与类型处理、标准化。

    Returns:
        (X, y, feature_names) 或 None（样本/特征不足时）。
        X、y 为 numpy 数组；feature_names 为最终使用的特征名列表。
    """
    covariates = covariates or []
    available_features = [f for f in feature_list if f in df.columns]
    if len(available_features) < len(covariates):
        return None

    X = df[available_features].copy()
    y = df[target].copy()

    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask].copy()
    y = y[mask].copy()

    if len(X) < min_samples:
        return None

    # 将 object 列转为数值（处理 '<0.4'、'>100' 等字符串）
    for col in X.columns:
        if X[col].dtype == object:
            s = X[col].astype(str).str.strip()
            s = s.str.replace(r"^<(\d*\.?\d+)$", r"\1", regex=True)
            s = s.str.replace(r"^>(\d*\.?\d+)$", r"\1", regex=True)
            X[col] = pd.to_numeric(s, errors="coerce")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in X.columns and X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

    X = X.dropna(axis=1)
    available_features = [f for f in available_features if f in X.columns]
    if len(available_features) < len(covariates):
        return None
    if len(X) < min_samples:
        return None

    X = X[available_features]
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X.values
    y = y.values.astype(int)

    return (X, y, available_features)
