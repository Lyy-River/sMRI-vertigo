# -*- coding: utf-8 -*-
"""
Configuration for clinical analysis pipeline.
Variable groups, covariates, quantiles, and optional file names.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Covariates (included in all models)
# ---------------------------------------------------------------------------
COVARIATES: List[str] = ["age", "sex"]

# Optional covariate (time diff between exam and MRI sequence)
TIME_DIFF_COL: Optional[str] = "abs_time_diff_days"

# ---------------------------------------------------------------------------
# Variable groups: raw names (before column-name normalization)
# ---------------------------------------------------------------------------
BIOCHEM_FEATURES_RAW: List[str] = [
    "CREA", "A/G", "APO-B", "eGFR", "AST/ALT", "URIC", "CKMB",
    "B:C", "CO2", "BUN", "LDH", "PHOS", "AST", "ALP",
    "DBIL", "GLO", "LDL-C", "HDLC", "GGTL", "GLU",
    "APO-A1", "ALB", "TBIL", "K", "TP", "Ca",
    "IBIL", "CK", "TG", "AG", "APOB:A",
    "ALT", "CL", "CHOL", "OSM", "Mg",
]

BLOOD_FEATURES: List[str] = [
    "BASO#", "BASO%", "EO#", "EO%", "HCT", "HGB", "LYM#", "LYM%", "MCH",
    "MCHC", "MCV", "MONO#", "MONO%", "MPV", "NEUT#", "NEUT%", "NRBC#",
    "NRBC%", "P-LCR", "PCT-XE", "PDW", "PLT", "RBC", "RDW-CV", "RDW-SD", "WBC",
]

ECG_FEATURES: List[str] = [
    "Sinus_rhythm", "Normal_ECG", "Rhythm_abnormality", "AV_block",
    "Bundle_branch_block", "Axis_abnormality", "Structural_pattern",
    "Repolarization_abnormality", "Ectopy", "Low_voltage_pattern",
]

DIAGNOSIS_FEATURES: List[str] = [
    "Certain_infectious_and_parasitic_diseases",
    "Codes_for_special_purposes",
    "Congenital_malformations_and_genetic_disorders",
    "Diseases_of_the_blood_and_blood-forming_organs_and_immune_mechanism_disorders",
    "Diseases_of_the_circulatory_system",
    "Diseases_of_the_digestive_system",
    "Diseases_of_the_ear_and_mastoid_process",
    "Diseases_of_the_genitourinary_system",
    "Diseases_of_the_musculoskeletal_system_and_connective_tissue",
    "Diseases_of_the_nervous_system",
    "Diseases_of_the_respiratory_system",
    "Endocrine,_nutritional_and_metabolic_diseases",
    "Factors_influencing_health_status",
    "Mental,_Behavioral_and_Neurodevelopmental_disorders",
    "Neoplasms",
    "Symptoms,_signs_and_abnormal_findings",
]

# ---------------------------------------------------------------------------
# Quantiles for quantile regression
# ---------------------------------------------------------------------------
QUANTILES: List[float] = [0.25, 0.50, 0.75]

# ---------------------------------------------------------------------------
# Clinical data file names (under clinical_dir)
# Use these keys in build_analysis_dataset. Override in get_config() if your filenames differ.
# ---------------------------------------------------------------------------
CLINICAL_FILE_NAMES: Dict[str, str] = {
    "covariates": "covariates.csv",
    "biochemical": "biochemical_final.csv",
    "cbc": "cbc_final.csv",
    "ecg": "ecg_structured.csv",
    "diagnosis": "diagnosis_final.csv",
}

# ---------------------------------------------------------------------------
# Cluster file: required columns
# ---------------------------------------------------------------------------
CLUSTER_ID_COL: str = "SubjectID"
CLUSTER_LABEL_COL: str = "Cluster"

# Columns that may contain covariates if no separate covariates file
CLUSTER_OPTIONAL_COVARIATE_COLS: List[str] = ["age", "sex", "sequence_time"]


def get_biochem_features_normalized() -> List[str]:
    """Return biochemical feature names after / - : replaced by _."""
    out: List[str] = []
    for name in BIOCHEM_FEATURES_RAW:
        out.append(name.replace("/", "_").replace("-", "_").replace(":", "_"))
    return out


def get_config(
    quantiles: Optional[List[float]] = None,
    time_diff_in_models: bool = True,
) -> Dict[str, Any]:
    """
    Return a config dict for the pipeline.
    """
    file_names = CLINICAL_FILE_NAMES
    return {
        "covariates": COVARIATES.copy(),
        "time_diff_col": TIME_DIFF_COL if time_diff_in_models else None,
        "quantiles": quantiles if quantiles is not None else QUANTILES.copy(),
        "clinical_file_names": file_names.copy(),
        "biochem_features_raw": BIOCHEM_FEATURES_RAW.copy(),
        "blood_features": BLOOD_FEATURES.copy(),
        "ecg_features": ECG_FEATURES.copy(),
        "diagnosis_features": DIAGNOSIS_FEATURES.copy(),
        "cluster_id_col": CLUSTER_ID_COL,
        "cluster_label_col": CLUSTER_LABEL_COL,
    }


# Default config
DEFAULT_CONFIG: Dict[str, Any] = get_config()
