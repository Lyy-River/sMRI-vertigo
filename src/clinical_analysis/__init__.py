# -*- coding: utf-8 -*-
"""
Clinical analysis pipeline: mean, distribution, and quantile differences across MRI-derived clusters.
"""

from . import config
from . import io
from . import ancova
from . import distribution_analysis
from . import quantile_models
from . import ml_nonlinearity
from . import logistic_models
from . import plotting
from . import pipeline

from .config import (
    DEFAULT_CONFIG,
    COVARIATES,
    QUANTILES,
    get_config,
)
from .io import build_analysis_dataset
from .pipeline import run_full_clinical_analysis
from .ancova import run_ancova_models
from .distribution_analysis import run_kde_visualization, run_ks_distribution_tests
from .quantile_models import run_quantile_models
from .ml_nonlinearity import run_rf_feature_selection
from .logistic_models import run_logistic_models

__all__ = [
    "config",
    "io",
    "ancova",
    "distribution_analysis",
    "quantile_models",
    "ml_nonlinearity",
    "logistic_models",
    "plotting",
    "pipeline",
    "DEFAULT_CONFIG",
    "COVARIATES",
    "QUANTILES",
    "get_config",
    "build_analysis_dataset",
    "run_full_clinical_analysis",
    "run_ancova_models",
    "run_kde_visualization",
    "run_ks_distribution_tests",
    "run_quantile_models",
    "run_rf_feature_selection",
    "run_logistic_models",
]
