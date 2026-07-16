# sMRI_pipeline

Publication code for **morphometric similarity network (MSN)** analysis of structural MRI and downstream **clinical statistical analyses** across MRI-derived patient subtypes.

This repository accompanies a Data Descriptor / Research Article. It is intended for reviewers and researchers who wish to inspect methods and reproduce analyses after obtaining the study data from the external archive linked below.

## Repository description (for Data Descriptor)

> Open-source analysis code for building morphometric similarity networks from regional morphometric features, clustering patients, performing network-based statistics, and relating MRI-derived clusters to clinical measures. Data are distributed separately; this repository contains Python packages, curated notebooks, and documentation for reproducible analysis.

## Pipeline overview

```text
External data (Zenodo / OSF / journal)
        |
        v
 msn_pipeline / msn_stats   -->  cluster labels, network stats
        |
        v
 clinical_analysis          -->  tables and figures for the paper
```

| Package | Role |
|---------|------|
| `msn_pipeline` | Feature loading, MSN construction, graph/GCN features, clustering |
| `msn_stats` | Edge-wise GLM, NBS, topology / between-group statistics |
| `clinical_analysis` | ANCOVA, KS, quantile regression, logistic models, plotting |
| `cluster_models` | Optional classifiers for cluster-related modeling |

## Requirements

- Python >= 3.10
- Core dependencies are listed in `pyproject.toml`
- Optional deep-learning extras (`torch`, `torch-geometric`) for GCN feature extraction: `pip install -e ".[msn]"`

## Installation

```bash
git clone https://github.com/REPLACE_ORG/sMRI_pipeline.git
cd sMRI_pipeline
pip install -e ".[dev]"
# Optional GCN support:
# pip install -e ".[msn]"
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate smri-pipeline
pip install -e ".[dev]"
```

## Data access

**No patient-level data are distributed in this repository.**

1. Download the study dataset from: **TODO: Zenodo / OSF DOI** (or the journal Data Availability statement).
2. Unpack files into `data/` following the layout in [`data/README.md`](data/README.md).
3. Confirm that feature matrices, clinical tables, and cluster label files are present.

## Minimal reproduction

After data are in place:

```bash
# MSN / clustering pipeline (edit scripts/example_msn_config.yaml as needed)
msn-pipeline --config scripts/example_msn_config.yaml

# Clinical pipeline for one cluster assignment file
clinical-pipeline \
  --cluster-file data/clusters/kmeans_k3.csv \
  --clinical-dir data/clinical_final \
  --output-dir results/clinical_kmeans_k3
```

Curated notebooks:

- [`notebooks/msn/`](notebooks/msn/) — MSN construction, clustering, NBS, topology
- [`notebooks/clinical/`](notebooks/clinical/) — clinical analyses by cluster solution

See [`docs/reproducibility.md`](docs/reproducibility.md) for mapping notebooks/scripts to paper figures and tables.

## Documentation

- [Pipeline overview](docs/overview.md)
- [MSN package notes](docs/msn_pipeline.md)
- [Clinical package notes](docs/clinical_analysis.md)
- [Reproducibility map](docs/reproducibility.md)

## Citation

Please cite this repository and the associated paper (placeholders until publication):

```bibtex
@software{smri_pipeline,
  title  = {sMRI_pipeline: MSN and clinical analysis code},
  author = {{REPLACE_WITH_AUTHOR_LIST}},
  year   = {2026},
  url    = {https://github.com/REPLACE_ORG/sMRI_pipeline},
  version = {0.1.0}
}
```

Also see [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).

## Contact

- Maintainer email: `REPLACE_WITH_AUTHOR_EMAIL@example.com`
- Issues: GitHub Issues on this repository

## What you still need to fill in before release

- [ ] Zenodo / OSF DOI and download URL in `data/README.md` and this README
- [ ] Author names, affiliations, and email in `pyproject.toml`, `CITATION.cff`, and README
- [ ] Paper title, journal, and preferred citation once accepted
- [ ] GitHub organization/username (`REPLACE_ORG`)
- [ ] Exact notebook ↔ figure/table mapping in `docs/reproducibility.md`
