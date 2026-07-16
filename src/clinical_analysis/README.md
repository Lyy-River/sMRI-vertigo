# clinical_analysis

Pipeline for analyzing clinical variables across MRI-derived clusters: **mean**, **distribution**, and **quantile** differences.

## Pipeline order

1. **Feature selection** — RF permutation importance (and LR vs RF CV comparison)
2. **Distribution exploration** — KDE plots by cluster
3. **Mean effect** — ANCOVA (Y ~ C(Cluster) + age + sex + time_diff)
4. **Distribution difference** — KS test (pairwise cluster comparison)
5. **Quantile effect** — Quantile regression (τ = 0.25, 0.50, 0.75)
6. **Binary outcomes** — Logistic regression (ECG, diagnosis)

## Usage

```python
from pathlib import Path
from clinical_analysis import run_full_clinical_analysis, build_analysis_dataset, get_config

# Folder structure: clinical_final + clusters
cluster_file = Path("data/用于分析data/clusters/cluster_k3.csv")
clinical_dir = Path("data/用于分析data/clinical_final")
output_dir = Path("results/cluster_k3")

results = run_full_clinical_analysis(
    cluster_file,
    clinical_dir,
    output_dir,
    config=None,
)
# Tables -> output_dir/tables/, figures -> output_dir/figures/
```

## Output tables

- `rf_cv_summary.csv` — LR vs RF CV accuracy, p-value, supports_nonlinear
- `rf_importance_*.csv` — Permutation importance by category
- `ancova_results.csv` — F, p_value, p_fdr per variable
- `ks_distribution_tests.csv` — variable, pair, ks_stat, p_value, p_fdr
- `quantile_regression_results.csv` — variable, quantile, cluster_effect_pvalue, p_fdr
- `logistic_results.csv` — Binary outcomes (ECG, diagnosis), p_value, p_fdr

## Config

Use `get_config(use_legacy_file_names=True)` or override `clinical_file_names` to point to your CSV filenames.
