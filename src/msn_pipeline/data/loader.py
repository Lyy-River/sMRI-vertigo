from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .paths import DataPaths


def load_subject_features(directory: str | Path) -> Dict[str, pd.DataFrame]:
    """Load per-subject feature CSVs from a directory into a dict.
    Subject ID 使用文件名 stem，并去掉末尾的 _feature_matrix 以便与 info 等表对齐。
    """
    base = Path(directory)
    subjects: Dict[str, pd.DataFrame] = {}
    for csv_path in base.glob("*.csv"):
        subject_id = csv_path.stem.removesuffix("_feature_matrix")
        subjects[subject_id] = pd.read_csv(csv_path, index_col=0)
    return subjects

