"""
MSN 网络边水平统计与 NBS 流程。

- 数据准备：加载 MSN 矩阵、协变量（age, sex, eTIV）、组/聚类标签
- 逐边 GLM（两组）：edge ~ group + age + sex + eTIV
- 逐边 ANCOVA（多组）：edge ~ cluster + age + sex + eTIV
- FDR 校正、事后两两比较、NBS 网络层推断
"""

from .data import (
    N_EDGES,
    N_NODES,
    load_nc_patient_data,
    load_patient_cluster_data,
    build_design_nc_patient,
    build_design_cluster,
)
from .edgewise import edgewise_glm_twogroup, edgewise_ancova_multigroup
from .fdr import fdr_correct
from .posthoc import pairwise_ttest_fdr, build_cluster_contrasts

__all__ = [
    "N_EDGES",
    "N_NODES",
    "load_nc_patient_data",
    "load_patient_cluster_data",
    "build_design_nc_patient",
    "build_design_cluster",
    "edgewise_glm_twogroup",
    "edgewise_ancova_multigroup",
    "fdr_correct",
    "pairwise_ttest_fdr",
    "build_cluster_contrasts",
]
