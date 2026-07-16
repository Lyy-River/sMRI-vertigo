# Pipeline overview

This repository implements two complementary analysis stages used in the associated paper:

1. **MSN stage** (`msn_pipeline`, `msn_stats`): build morphometric similarity networks from regional morphometric feature matrices, extract graph / GCN features, cluster patients, and run network statistics (including NBS).
2. **Clinical stage** (`clinical_analysis`, `cluster_models`): merge MRI-derived cluster labels with cleaned clinical tables and run mean / distribution / quantile / logistic analyses.

```text
External data (Zenodo/OSF)
        |
        v
  Feature matrices  --->  msn_pipeline  --->  cluster labels
        |                      |                    |
        |                      v                    v
        |                 msn_stats            clinical_analysis
        |              (NBS / topology)         (ANCOVA, KS, ...)
        |                      |                    |
        +----------------------+--------------------+
                               |
                               v
                     Tables and figures for the paper
```

See also:

- [msn_pipeline.md](msn_pipeline.md)
- [clinical_analysis.md](clinical_analysis.md)
- [reproducibility.md](reproducibility.md)
