# -*- coding: utf-8 -*-
"""各模型工厂函数：LR、RF、XGBoost、SVM、KNN，统一 get_model(name, **params) 入口。"""

from typing import Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


def get_lr(
    *,
    multi_class: str = "multinomial",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    C: float = 1.0,
    class_weight: str = "balanced",
    random_state: int = 42,
    **kwargs: Any,
) -> LogisticRegression:
    """多分类逻辑回归。"""
    return LogisticRegression(
        multi_class=multi_class,
        solver=solver,
        max_iter=max_iter,
        C=C,
        class_weight=class_weight,
        random_state=random_state,
        **kwargs,
    )


def get_rf(
    *,
    n_estimators: int = 500,
    max_depth: int = 8,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    class_weight: str = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestClassifier:
    """随机森林多分类。"""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )


def get_xgb(
    *,
    n_classes: int = 3,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    **kwargs: Any,
):
    """XGBoost 多分类。需要安装 xgboost。"""
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost is required for get_xgb(). Install with: pip install xgboost")
    return xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=n_classes,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        **kwargs,
    )


def get_svm(
    *,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    class_weight: str = "balanced",
    decision_function_shape: str = "ovr",
    random_state: int = 42,
    **kwargs: Any,
) -> SVC:
    """SVM 多分类（RBF 核）。"""
    return SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        decision_function_shape=decision_function_shape,
        random_state=random_state,
        **kwargs,
    )


def get_knn(
    *,
    n_neighbors: int = 15,
    weights: str = "uniform",
    **kwargs: Any,
) -> KNeighborsClassifier:
    """K 近邻多分类。"""
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, **kwargs)


_MODEL_FACTORIES = {
    "lr": get_lr,
    "rf": get_rf,
    "xgb": get_xgb,
    "svm": get_svm,
    "knn": get_knn,
}


def get_model(name: str, n_classes: int = 3, **params: Any):
    """
    按名字返回已配置的 estimator，供 cross_val_score 等使用。

    name: 'lr' | 'rf' | 'xgb' | 'svm' | 'knn'
    n_classes: 类别数（仅 xgb 使用）。
    params: 传给对应 get_* 的关键字参数。
    """
    name = name.lower().strip()
    if name not in _MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {name}. Choose from {list(_MODEL_FACTORIES.keys())}")
    factory = _MODEL_FACTORIES[name]
    if name == "xgb":
        return factory(n_classes=n_classes, **params)
    return factory(**params)
