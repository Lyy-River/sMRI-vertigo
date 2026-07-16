# Data (external)

**No patient-level or identifiable clinical/MRI data are distributed in this GitHub repository.**

Analysis data are provided separately (Zenodo / OSF / journal Data Descriptor). After download, unpack them so that the repository root looks like the layout below.

## Download

| Resource | URL / DOI |
|----------|-----------|
| Primary data package | `TODO: replace with Zenodo or OSF DOI` |
| Paper / Data Descriptor | `TODO: replace with article DOI or preprint URL` |

If links are not yet public, contact the corresponding author listed in `CITATION.cff`.

## Expected directory layout

Place files under `data/` (paths relative to the repository root):

```text
data/
  README.md                          # this file (tracked in git)
  health_feature_matrices/           # per-subject CSV feature matrices (controls)
    subXXXXX_feature_matrix.csv
  patient_feature_matrices/          # per-subject CSV feature matrices (patients)
    subXXXXX_feature_matrix.csv
  clinical_final/                    # cleaned clinical tables used by clinical_analysis
    covariates.csv                   # SubjectID, age, sex, sequence_time, ...
    biochemical_final.csv
    cbc_final.csv
    ecg_structured.csv
    diagnosis_final.csv
  clusters/                          # MRI-derived cluster assignments
    kmeans_k3.csv                    # SubjectID, Cluster
    kmeans_k2.csv
    spectral_k2.csv
    ...
```

## File conventions

### Feature matrices (MSN)

- One CSV per subject.
- Rows: brain regions; columns: morphometric features (as used in the paper).
- Subject IDs in filenames should match IDs used in cluster and clinical tables.

### Cluster assignment CSVs

Minimum columns:

- `SubjectID`
- `Cluster` (integer or categorical label)

### Clinical tables (`clinical_final/`)

- Join key: `SubjectID`.
- Covariates are read from `covariates.csv` only (cluster files must not redefine age/sex).
- Exact filenames can be overridden via `clinical_analysis.config.get_config(...)`.

## Outputs (local only)

Generated results should be written under `results/` (gitignored). Do not commit CSV/XLSX/NIfTI outputs.
