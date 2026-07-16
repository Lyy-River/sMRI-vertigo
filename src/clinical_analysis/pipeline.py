# -*- coding: utf-8 -*-
"""
Unified pipeline: load data, run all analyses, export tables and figures.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from . import config as _config
from . import io as _io
from . import ancova
from . import distribution_analysis
from . import quantile_models
from . import ml_nonlinearity
from . import logistic_models
from . import plotting
import pandas as pd


def run_full_clinical_analysis(
    cluster_file: Path,
    clinical_dir: Path,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    *,
    run_feature_selection: bool = True,
    run_kde: bool = True,
    run_ancova: bool = True,
    run_ks: bool = True,
    run_quantile: bool = True,
    run_logistic: bool = True,
    export_tables: bool = True,
    export_figures: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline:

    1. Load and merge datasets (build_analysis_dataset)
    2. Feature selection (RF importance, optional LR vs RF comparison)
    3. Distribution exploration (KDE plots)
    4. Mean effect testing (ANCOVA)
    5. Distribution difference testing (KS test)
    6. Quantile regression
    7. (Optional) Logistic models for binary outcomes
    8. Save tables and figures to output_dir

    cluster_file: path to cluster assignment CSV (SubjectID, Cluster).
    clinical_dir: directory with clinical_final CSVs.
    output_dir: where to write results.
    config: optional config dict; if None, uses DEFAULT_CONFIG.
    """
    if config is None:
        config = _config.get_config()

    # 1) Build datasets
    datasets = _io.build_analysis_dataset(
        cluster_file, clinical_dir, config=config
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    if export_tables:
        tables_dir.mkdir(parents=True, exist_ok=True)
    if export_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {"datasets": datasets, "config": config}

    # 2) Feature selection (RF + LR vs RF)
    if run_feature_selection:
        rf_out = ml_nonlinearity.run_rf_feature_selection(datasets, config=config)
        results["rf_summary"] = rf_out["summary"]
        results["rf_importance"] = rf_out["importance"]
        if export_tables and rf_out["summary"]:
            pd.DataFrame(rf_out["summary"]).to_csv(tables_dir / "rf_cv_summary.csv", index=False)
            for cat, imp_df in rf_out["importance"].items():
                imp_df.to_csv(tables_dir / f"rf_importance_{cat}.csv", index=False)

    # 3) KDE
    if run_kde:
        kde_figs = distribution_analysis.run_kde_visualization(
            datasets, config=config, output_dir=figures_dir if export_figures else None, top_n_per_group=6
        )
        results["kde_figures"] = kde_figs

    # 4) ANCOVA
    if run_ancova:
        ancova_df = ancova.run_ancova_models(datasets, config=config)
        results["ancova"] = ancova_df
        if export_tables and not ancova_df.empty:
            ancova_df.to_csv(tables_dir / "ancova_results.csv", index=False)

    # 5) KS
    if run_ks:
        ks_df = distribution_analysis.run_ks_distribution_tests(datasets, config=config)
        results["ks_tests"] = ks_df
        if export_tables and not ks_df.empty:
            ks_df.to_csv(tables_dir / "ks_distribution_tests.csv", index=False)

    # 6) Quantile regression
    if run_quantile:
        quantile_df = quantile_models.run_quantile_models(datasets, config=config)
        results["quantile"] = quantile_df
        if export_tables and not quantile_df.empty:
            quantile_df.to_csv(tables_dir / "quantile_regression_results.csv", index=False)

    # 7) Logistic (ECG, diagnosis)
    if run_logistic:
        logit_df = logistic_models.run_logistic_models(datasets, config=config)
        results["logistic"] = logit_df
        if export_tables and not logit_df.empty:
            logit_df.to_csv(tables_dir / "logistic_results.csv", index=False)

    return results
