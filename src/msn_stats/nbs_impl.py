"""
GLM-based Network-Based Statistic (NBS) for morphometric similarity networks.

Migrated from src/nbs_glm.py. Implements:
- Manual OLS via matrix algebra (no sklearn)
- Edge-wise t-statistics for a specified contrast
- Primary threshold on t-map; connected components via networkx
- Freedman-Lane permutation (FWER-corrected p-values using max component size)

Reference: Zalesky et al. (2010) NBS. For Freedman-Lane: Freedman & Lane (1983).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import networkx as nx


def _n_edges(n_nodes: int) -> int:
    return n_nodes * (n_nodes - 1) // 2


def matrix_to_edges(A: np.ndarray) -> np.ndarray:
    """Vectorize upper triangle of symmetric matrix. (n,n) or (n_subj,n,n) -> (n_edges,) or (n_subj,n_edges)."""
    if A.ndim == 2:
        n = A.shape[0]
        if A.shape[1] != n:
            raise ValueError("matrix_to_edges: matrix must be square")
        return A[np.triu_indices(n, k=1)]
    if A.ndim == 3:
        n_subj, n, _ = A.shape
        if A.shape[2] != n:
            raise ValueError("matrix_to_edges: matrices must be square")
        E = _n_edges(n)
        out = np.empty((n_subj, E), dtype=A.dtype)
        for i in range(n_subj):
            out[i] = A[i][np.triu_indices(n, k=1)]
        return out
    raise ValueError("matrix_to_edges: A must be 2D or 3D")


def edges_to_matrix(edges: np.ndarray, n_nodes: int) -> np.ndarray:
    """Fill upper triangle from edge vector; lower triangle mirrored. (n_edges,) -> (n, n)."""
    E = _n_edges(n_nodes)
    if edges.size != E:
        raise ValueError(f"edges size {edges.size} != n_edges {E}")
    A = np.zeros((n_nodes, n_nodes), dtype=edges.dtype)
    triu = np.triu_indices(n_nodes, k=1)
    A[triu] = edges
    A.T[triu] = edges
    return A


def edge_index_to_ij(edge_idx: int, n_nodes: int) -> Tuple[int, int]:
    """Map linear edge index to (i, j) with i < j."""
    i = 0
    start = 0
    while i < n_nodes:
        count = n_nodes - 1 - i
        if edge_idx < start + count:
            j = i + 1 + (edge_idx - start)
            return (i, j)
        start += count
        i += 1
    raise IndexError("edge_idx out of range")


def ij_to_edge_index(i: int, j: int, n_nodes: int) -> int:
    """(i, j) with i < j -> linear edge index."""
    if i >= j:
        raise ValueError("i must be < j")
    return i * (2 * n_nodes - 1 - i) // 2 + (j - i - 1)


def _ols_fit(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full model Y = X @ B + E. Returns B (p, E), R (n, E), MSE (E,), iXtX."""
    n, p = X.shape
    E = Y.shape[1]
    XtX = X.T @ X
    iXtX = np.linalg.inv(XtX)
    XtY = X.T @ Y
    B = iXtX @ XtY
    R = Y - X @ B
    df = n - p
    MSE = np.sum(R * R, axis=0) / np.maximum(df, 1)
    return B, R, MSE, iXtX


def _edgewise_t(
    B: np.ndarray,
    R: np.ndarray,
    MSE: np.ndarray,
    X: np.ndarray,
    iXtX: np.ndarray,
    contrast: np.ndarray,
) -> np.ndarray:
    """t-statistic for contrast c'B per edge."""
    c = np.asarray(contrast, dtype=np.float64).ravel()
    if c.ndim != 1 or c.size != B.shape[0]:
        raise ValueError("contrast must be 1D of length p")
    effect = c @ B
    c_iXtX_c = float(c @ iXtX @ c)
    var_effect = c_iXtX_c * MSE
    se = np.sqrt(np.maximum(var_effect, np.finfo(float).tiny))
    return effect / se


def _null_design(X: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    """Design matrix under H0: c'B = 0. Returns X0 (n, p0) with p0 = p-1."""
    c = np.asarray(contrast, dtype=np.float64).ravel()
    n, p = X.shape
    if c.size != p:
        raise ValueError("contrast length must equal number of columns")
    nz = np.flatnonzero(c)
    if nz.size == 1:
        j = int(nz[0])
        return np.delete(X, j, axis=1)
    U, s, Vh = np.linalg.svd(c.reshape(1, -1))
    A = Vh[1:].T
    return X @ A


def _thresholded_components(
    t_map: np.ndarray,
    n_nodes: int,
    threshold: float,
    two_tailed: bool,
) -> Tuple[nx.Graph, List[np.ndarray]]:
    """Build graph from edges with |t| >= threshold; return graph and list of component edge masks."""
    E = t_map.size
    if E != _n_edges(n_nodes):
        raise ValueError("t_map size does not match n_nodes")
    suprathreshold = np.abs(t_map) >= threshold if two_tailed else t_map >= threshold
    G = nx.Graph()
    for idx in np.flatnonzero(suprathreshold):
        i, j = edge_index_to_ij(int(idx), n_nodes)
        G.add_edge(i, j)
    comps = list(nx.connected_components(G))
    component_masks = []

    for comp in comps:
        mask = np.zeros(E, dtype=bool)
        for i, j in G.edges():
            if i in comp and j in comp:
                # NetworkX 的无向图边 (i, j) 不保证 i < j，这里强制排序后再映射到上三角索引
                a, b = (i, j) if i < j else (j, i)
                mask[ij_to_edge_index(a, b, n_nodes)] = True
        component_masks.append(mask)

    return G, component_masks


def _component_sizes(component_masks: List[np.ndarray]) -> np.ndarray:
    return np.array([m.sum() for m in component_masks], dtype=np.intp)


@dataclass
class NBSResult:
    """Results of GLM-based NBS."""

    t_map: np.ndarray
    component_masks: List[np.ndarray]
    component_pvalues: np.ndarray
    component_sizes: np.ndarray
    n_components: int
    threshold: float
    n_perm: int
    n_nodes: int


def nbs_glm(
    edge_matrix: np.ndarray,
    design: np.ndarray,
    contrast: np.ndarray,
    n_nodes: int,
    threshold: float,
    n_perm: int = 5000,
    two_tailed: bool = True,
    seed: Optional[int] = None,
) -> NBSResult:
    """
    GLM-based NBS with Freedman-Lane permutation.

    edge_matrix: (n_subjects, n_edges) or (n_subjects, n_nodes, n_nodes).
    design: (n_subjects, n_predictors), include intercept.
    contrast: (n_predictors,) e.g. [0, 1, 0].
    """
    rng = np.random.default_rng(seed)
    if edge_matrix.ndim == 3:
        edge_matrix = matrix_to_edges(edge_matrix)
    n_subj, n_edges = edge_matrix.shape
    expected_edges = _n_edges(n_nodes)
    if n_edges != expected_edges:
        raise ValueError(f"n_edges {n_edges} != expected {expected_edges} for n_nodes={n_nodes}")
    if design.shape[0] != n_subj:
        raise ValueError("design row count must match number of subjects")
    n_pred = design.shape[1]
    if np.size(contrast) != n_pred:
        raise ValueError("contrast length must match number of predictors")

    X = np.asarray(design, dtype=np.float64)
    Y = np.asarray(edge_matrix, dtype=np.float64)
    contrast = np.asarray(contrast, dtype=np.float64).ravel()

    B, R, MSE, iXtX = _ols_fit(X, Y)
    t_obs = _edgewise_t(B, R, MSE, X, iXtX, contrast)

    X0 = _null_design(X, contrast)
    X0tX0 = X0.T @ X0
    iX0tX0 = np.linalg.inv(X0tX0)
    B0 = iX0tX0 @ (X0.T @ Y)
    R0 = Y - X0 @ B0
    X0B0 = X0 @ B0

    c_iXtX_c = float(contrast @ iXtX @ contrast)
    df = n_subj - n_pred

    G_obs, comp_masks = _thresholded_components(t_obs, n_nodes, threshold, two_tailed)
    sizes_obs = _component_sizes(comp_masks)
    max_size_obs = int(sizes_obs.max()) if sizes_obs.size else 0

    count_max_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n_subj)
        R0_perm = R0[perm]
        Y_star = X0B0 + R0_perm
        B_star = iXtX @ (X.T @ Y_star)
        R_star = Y_star - X @ B_star
        MSE_star = np.sum(R_star * R_star, axis=0) / np.maximum(df, 1)
        effect_star = contrast @ B_star
        var_effect = c_iXtX_c * MSE_star
        se_star = np.sqrt(np.maximum(var_effect, np.finfo(float).tiny))
        t_star = effect_star / se_star
        _, comp_masks_star = _thresholded_components(t_star, n_nodes, threshold, two_tailed)
        sizes_star = _component_sizes(comp_masks_star)
        max_size_star = int(sizes_star.max()) if sizes_star.size else 0
        if max_size_star >= max_size_obs:
            count_max_ge += 1

    fwer_p = (count_max_ge + 1) / (n_perm + 1)
    comp_pvalues = np.full(len(comp_masks), fwer_p, dtype=np.float64)

    return NBSResult(
        t_map=t_obs,
        component_masks=comp_masks,
        component_pvalues=comp_pvalues,
        component_sizes=sizes_obs,
        n_components=len(comp_masks),
        threshold=threshold,
        n_perm=n_perm,
        n_nodes=n_nodes,
    )


def component_mask_to_matrix(mask: np.ndarray, n_nodes: int) -> np.ndarray:
    """Convert (n_edges,) boolean component mask to (n_nodes, n_nodes) symmetric matrix."""
    E = mask.size
    if E != _n_edges(n_nodes):
        raise ValueError("mask size does not match n_nodes")
    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    triu = np.triu_indices(n_nodes, k=1)
    A[triu] = np.where(mask, 1.0, 0.0)
    A.T[triu] = A[triu]
    return A


def nbs_glm_from_matrices(
    matrices: np.ndarray,
    design: np.ndarray,
    contrast: np.ndarray,
    n_nodes: int,
    threshold: float,
    n_perm: int = 5000,
    two_tailed: bool = True,
    seed: Optional[int] = None,
) -> NBSResult:
    """Convenience: (n_subjects, n_nodes, n_nodes) -> vectorize and run nbs_glm."""
    edge_matrix = matrix_to_edges(matrices)
    return nbs_glm(
        edge_matrix=edge_matrix,
        design=design,
        contrast=contrast,
        n_nodes=n_nodes,
        threshold=threshold,
        n_perm=n_perm,
        two_tailed=two_tailed,
        seed=seed,
    )
