# Notebooks

English-named notebooks for reproducing the analysis workflow.

## Prerequisites

1. Install the package: `pip install -e ".[dev]"` (add `[msn]` for GCN features).
2. Download external data and place it under `data/` as described in [`data/README.md`](../data/README.md).
3. Run notebooks from the repository root (or ensure `Path("data")` resolves correctly).

## MSN (`notebooks/msn/`)

| Notebook | Purpose |
|----------|---------|
| `01_build_health_msn.ipynb` | Build morphometric similarity networks for healthy controls |
| `02_build_patient_msn.ipynb` | Build MSN for patients |
| `03_clustering.ipynb` | Subject clustering from graph / GCN features |
| `04_nc_vs_patient_network.ipynb` | Between-group network comparisons |
| `05_nbs_analysis.ipynb` | Network-based statistic (NBS) analyses |
| `06_gcn_features.ipynb` | Extract and save GCN-based features |
| `07_topology_statistics.ipynb` | Topology / graph metric statistics |

## Clinical (`notebooks/clinical/`)

| Notebook | Purpose |
|----------|---------|
| `01_clinical_pipeline.ipynb` | End-to-end clinical analysis for a cluster assignment |
| `02_cluster_kmeans3.ipynb` | Clinical pipeline for k-means k=3 labels |
| `03_cluster_kmeans2.ipynb` | Clinical pipeline for k-means k=2 labels |
| `04_cluster_spectral2.ipynb` | Clinical pipeline for spectral k=2 labels |
| `05_cluster_average2.ipynb` | Clinical pipeline for average-linkage k=2 labels |

Heavy cell outputs were cleared to keep the repository lightweight. Re-run notebooks locally after placing data under `data/`.
