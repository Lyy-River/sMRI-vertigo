from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from msn_pipeline.msn import build_msn


class GCNFeatureExtractor(nn.Module):
    """GCN-based feature extractor for MSN graphs."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


def extract_patient_features(
    df_patient: pd.DataFrame,
    model: nn.Module,
    threshold: Optional[float] = None,
    target_density: float = 0.2,
    metric: str = "pearson",
) -> np.ndarray:
    """Extract GCN features from a single patient's MSN (DataFrame input)."""
    sim_matrix, G, regions = build_msn(
        df_patient,
        metric=metric,
        threshold=threshold,
        target_density=target_density,
    )

    node_features = torch.tensor(df_patient.values, dtype=torch.float)
    edges_list = list(G.edges())
    if len(edges_list) == 0:
        raise ValueError("Graph is empty. Try lowering threshold.")

    edge_index = torch.tensor(
        [[df_patient.index.get_loc(e[0]), df_patient.index.get_loc(e[1])] for e in edges_list],
        dtype=torch.long,
    ).t().contiguous()

    pyg_graph = Data(x=node_features, edge_index=edge_index)
    model.eval()
    with torch.no_grad():
        batch = torch.zeros(pyg_graph.num_nodes, dtype=torch.long)
        features = model(pyg_graph.x, pyg_graph.edge_index, batch)
        return features.cpu().numpy().flatten()


def extract_patient_features_from_file(
    file_path: Union[str, Path],
    model: nn.Module,
    threshold: Optional[float] = None,
    target_density: float = 0.2,
    metric: str = "pearson",
) -> np.ndarray:
    """Load subject CSV and extract GCN features (convenience wrapper)."""
    df = pd.read_csv(file_path, index_col=0)
    return extract_patient_features(
        df, model, threshold=threshold, target_density=target_density, metric=metric
    )


def extract_all_patients(
    input_dir: Union[str, Path],
    output_file: Union[str, Path],
    in_channels: int = 7,
    hidden_channels: int = 64,
    out_channels: int = 32,
    threshold: Optional[float] = None,
    target_density: float = 0.2,
    metric: str = "pearson",
) -> pd.DataFrame:
    """
    Batch extract GCN features for all subject CSVs in a directory and save to CSV.

    Returns DataFrame with columns Patient_ID, Feature_1, ..., Feature_N.
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    model = GCNFeatureExtractor(in_channels, hidden_channels, out_channels)

    patient_ids: list[str] = []
    patient_features: list[np.ndarray] = []

    for csv_path in sorted(input_dir.glob("*.csv")):
        try:
            feats = extract_patient_features_from_file(
                csv_path, model, threshold=threshold, target_density=target_density, metric=metric
            )
            patient_ids.append(csv_path.stem.removesuffix("_feature_matrix"))
            patient_features.append(feats)
        except Exception as e:
            print(f"{csv_path.stem} failed: {e}")

    out_df = pd.DataFrame(
        patient_features,
        columns=[f"Feature_{i + 1}" for i in range(out_channels)],
    )
    out_df.insert(0, "Patient_ID", patient_ids)
    out_df.to_csv(output_file, index=False)
    return out_df

