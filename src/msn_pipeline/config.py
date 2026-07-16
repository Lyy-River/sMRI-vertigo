from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PipelineConfig:
    data_root: Path
    experiment_name: str
    output_root: Path
    seed: int = 42

    msn_metric: str = "pearson"
    msn_target_density: float = 0.2

    clustering_method: str = "kmeans"
    clustering_n_clusters: int = 3

    stats_method: str = "nbs"


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load YAML configuration into a strongly-typed config object."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    general = raw.get("general", {})
    msn = raw.get("msn", {})
    clustering = raw.get("clustering", {})
    stats = raw.get("stats", {})

    return PipelineConfig(
        data_root=Path(general.get("data_root", "data")),
        experiment_name=general.get("experiment_name", "default_experiment"),
        output_root=Path(general.get("output_root", "results")),
        seed=int(general.get("seed", 42)),
        msn_metric=msn.get("metric", "pearson"),
        msn_target_density=float(msn.get("target_density", 0.2)),
        clustering_method=clustering.get("method", "kmeans"),
        clustering_n_clusters=int(clustering.get("n_clusters", 3)),
        stats_method=stats.get("method", "nbs"),
    )

