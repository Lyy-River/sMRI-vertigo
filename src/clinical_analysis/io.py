# -*- coding: utf-8 -*-
"""
I/O and dataset construction: read data, merge clinical datasets with cluster labels.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import config as _config


def _normalize_biochem_columns(df: pd.DataFrame, raw_list: List[str]) -> List[str]:
    """Return list of column names that exist in df after normalizing raw_list."""
    out: List[str] = []
    for name in raw_list:
        renamed = name.replace("/", "_").replace("-", "_").replace(":", "_")
        if renamed in df.columns:
            out.append(renamed)
    return out



def _ensure_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    """Coerce object columns to numeric (e.g. '<10' -> 10)."""
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].dtype == object or df[col].dtype.name == "object":
            s = df[col].astype(str).str.strip()
            s = s.str.replace(r"^<(\d*\.?\d+)$", r"\1", regex=True)
            s = s.str.replace(r"^>(\d*\.?\d+)$", r"\1", regex=True)
            df[col] = pd.to_numeric(s, errors="coerce")


def _read_optional_csv(clinical_dir: Path, key: str, file_names: Dict[str, str]) -> Optional[pd.DataFrame]:
    path = clinical_dir / file_names[key]
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_analysis_dataset(
    cluster_path: Path,
    clinical_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load cluster assignments and clinical datasets, merge on SubjectID.

    cluster_path: path to CSV with at least SubjectID and Cluster.
    clinical_dir: directory containing covariates and clinical CSVs (see config clinical_file_names).
    config: optional config dict; if None, uses DEFAULT_CONFIG.

    Returns dict with:
      - cluster_info: DataFrame with SubjectID, Cluster, age, sex (and optional time cols)
      - biochem: merged biochem DataFrame (or None)
      - cbc: merged blood/cbc DataFrame (or None)
      - ecg: merged ECG DataFrame (or None)
      - diagnosis: merged diagnosis DataFrame (or None)
      - config: the config used (feature lists resolved to existing columns)
    """
    if config is None:
        config = _config.get_config()
    file_names = config.get("clinical_file_names", _config.CLINICAL_FILE_NAMES)
    id_col = config.get("cluster_id_col", _config.CLUSTER_ID_COL)
    label_col = config.get("cluster_label_col", _config.CLUSTER_LABEL_COL)
    covars = config.get("covariates", _config.COVARIATES)

    # 1) Load cluster file: ONLY SubjectID + Cluster (no covariates here)
    cluster_df = pd.read_csv(cluster_path)
    if label_col not in cluster_df.columns:
        raise ValueError(f"Cluster file must contain column '{label_col}'.")
    if id_col not in cluster_df.columns:
        raise ValueError(f"Cluster file must contain column '{id_col}'.")
    cluster_df = cluster_df[[id_col, label_col]].copy()
    cluster_df[label_col] = cluster_df[label_col].astype(int)

    # 2) Covariates: ALWAYS from clinical_dir / covariates.csv
    covar_df = _read_optional_csv(clinical_dir, "covariates", file_names)
    if covar_df is None:
        raise FileNotFoundError(
            f"Covariates file '{file_names.get('covariates')}' not found in {clinical_dir}."
        )

    missing_covars = [c for c in covars if c not in covar_df.columns]
    if missing_covars:
        raise ValueError(
            f"Covariates file must contain columns {covars}, missing: {missing_covars}."
        )

    covar_cols = [id_col] + covars
    covar_df = covar_df[covar_cols].drop_duplicates(id_col)

    # type cleaning for covariates
    if "age" in covar_df.columns:
        covar_df["age"] = pd.to_numeric(covar_df["age"], errors="coerce")
    if "sex" in covar_df.columns:
        covar_df["sex"] = covar_df["sex"].astype(int)
    covar_df = covar_df.dropna(subset=covars)

    base = cluster_df.merge(covar_df, on=id_col, how="inner")
    base = base.dropna(subset=[label_col] + covars)

    base = base.dropna(subset=[label_col])
    datasets: Dict[str, Any] = {
        "cluster_info": base,
        "biochem": None,
        "cbc": None,
        "ecg": None,
        "diagnosis": None,
        "config": config,
    }

    # 3) Biochemical
    biochem_path = clinical_dir / file_names["biochemical"]
    if biochem_path.exists():
        biochem = pd.read_csv(biochem_path)
        biochem_cols = [id_col] + [c for c in biochem.columns if c != id_col and c != label_col]
        biochem = base[[id_col, label_col] + [c for c in covars if c in base.columns]].merge(
            biochem[biochem_cols],
            on=id_col,
            how="inner",
            suffixes=("", "_r"),
        )
        biochem = biochem[[c for c in biochem.columns if not c.endswith("_r")]]
        feats = _normalize_biochem_columns(biochem, config.get("biochem_features_raw", _config.BIOCHEM_FEATURES_RAW))
        config["biochem_features"] = feats
        datasets["biochem"] = biochem

    # 4) CBC / blood
    cbc_path = clinical_dir / file_names["cbc"]
    if cbc_path.exists():
        cbc = pd.read_csv(cbc_path)
        blood_feats = config.get("blood_features", _config.BLOOD_FEATURES)
        blood_feats = [f for f in blood_feats if f in cbc.columns]
        config["blood_features"] = blood_feats
        cbc_cols = [id_col] + blood_feats + [c for c in covars if c in cbc.columns] + (["abs_time_diff_days"] if "abs_time_diff_days" in cbc.columns else [])
        cbc_cols = [c for c in cbc_cols if c in cbc.columns]
        cbc_merged = base[[id_col, label_col] + [c for c in covars if c in base.columns]].merge(
            cbc[cbc_cols],
            on=id_col,
            how="inner",
            suffixes=("", "_r"),
        )
        cbc_merged = cbc_merged[[c for c in cbc_merged.columns if not c.endswith("_r")]]
        datasets["cbc"] = cbc_merged

    # 5) ECG（CSV 列名已与 config 一致，只取实际存在的列）
    ecg_path = clinical_dir / file_names["ecg"]
    if ecg_path.exists():
        ecg = pd.read_csv(ecg_path)
        ecg_feats = [f for f in config.get("ecg_features", _config.ECG_FEATURES) if f in ecg.columns]
        config["ecg_features"] = ecg_feats
        ecg_cols = [id_col] + ecg_feats + [c for c in covars if c in ecg.columns] + (["abs_time_diff_days"] if "abs_time_diff_days" in ecg.columns else [])
        ecg_cols = [c for c in ecg_cols if c in ecg.columns]
        ecg_merged = base[[id_col, label_col] + [c for c in covars if c in base.columns]].merge(
            ecg[ecg_cols],
            on=id_col,
            how="inner",
            suffixes=("", "_r"),
        )
        ecg_merged = ecg_merged[[c for c in ecg_merged.columns if not c.endswith("_r")]]
        datasets["ecg"] = ecg_merged

    # 6) Diagnosis（列名中任意空白统一为单下划线，与 config 一致）
    diag_path = clinical_dir / file_names["diagnosis"]
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        diag.columns = [re.sub(r"\s+", "_", str(c).strip()) for c in diag.columns]
        diag_feats = [f for f in config.get("diagnosis_features", _config.DIAGNOSIS_FEATURES) if f in diag.columns]
        config["diagnosis_features"] = diag_feats
        diag_cols = [id_col] + diag_feats + [c for c in covars if c in diag.columns] + (["abs_time_diff_days"] if "abs_time_diff_days" in diag.columns else [])
        diag_cols = [c for c in diag_cols if c in diag.columns]
        diag_merged = base[[id_col, label_col] + [c for c in covars if c in base.columns]].merge(
            diag[diag_cols],
            on=id_col,
            how="inner",
            suffixes=("", "_r"),
        )
        diag_merged = diag_merged[[c for c in diag_merged.columns if not c.endswith("_r")]]
        datasets["diagnosis"] = diag_merged

    return datasets
