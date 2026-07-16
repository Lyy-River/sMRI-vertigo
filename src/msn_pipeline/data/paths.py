from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataPaths:
    root: Path

    def patient_csv_dir(self) -> Path:
        return self.root / "patient_feature_matrices"

    def control_csv_dir(self) -> Path:
        return self.root / "health_feature_matrices"

