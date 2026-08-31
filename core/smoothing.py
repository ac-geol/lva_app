"""Graph smoothing, on the score field and on the tensor field.

Two separate jobs:

  smooth_scores      -- one or more passes over the neighbourhood graph before
     differencing. The structure tensor takes finite differences of the score,
     which amplifies nugget noise; assay data at 2 m support is mostly nugget.
     Smoothing first is the difference between estimating a gradient and
     estimating a sampling error.

  smooth_tensor_field -- regularizes the OUTPUT field. Minimizes
     sum_i ||T_i - T_i_obs||^2 + mu * sum_ij w_ij ||T_i - T_j||^2 by Jacobi
     iteration. Operating on tensors rather than eigenvectors sidesteps sign
     ambiguity completely: there is no flip to fix because tensors do not have
     one.
"""
from __future__ import annotations

import numpy as np

from .neighbors import EdgeSet


def _neighbour_average(values: np.ndarray, edges: EdgeSet) -> tuple:
    """Weighted mean of neighbour values per node, plus the weight totals."""
    n = edges.n_nodes
    wsum = np.bincount(edges.i, weights=edges.w, minlength=n)
    flat = values.reshape(len(values), -1)
    acc = np.stack([
        np.bincount(edges.i, weights=edges.w * flat[edges.j, k], minlength=n)
        for k in range(flat.shape[1])], axis=1)
    safe = np.where(wsum > 0, wsum, 1.0)
    return (acc / safe[:, None]).reshape(values.shape), wsum


def smooth_scores(scores: np.ndarray, edges: EdgeSet, *, iterations: int = 1,
                  alpha: float = 0.5) -> np.ndarray:
    """alpha=0 leaves scores untouched; alpha=1 replaces each with its
    neighbourhood mean. Nodes with no neighbours keep their own value."""
    out = np.asarray(scores, dtype=float).copy()
    for _ in range(max(0, iterations)):
        avg, wsum = _neighbour_average(out, edges)
        has = wsum > 0
        out[has] = (1.0 - alpha) * out[has] + alpha * avg[has]
    return out


def smooth_tensor_field(T: np.ndarray, edges: EdgeSet, *, mu: float = 1.0,
                        iterations: int = 10) -> np.ndarray:
    """Jacobi solve of the regularized tensor field. mu=0 is a no-op."""
    if mu <= 0 or iterations <= 0:
        return T
    obs = T
    cur = T.copy()
    n = edges.n_nodes
    wsum = np.bincount(edges.i, weights=edges.w, minlength=n)
    denom = (1.0 + mu * wsum)[:, None, None]
    for _ in range(iterations):
        avg, _ = _neighbour_average(cur, edges)
        cur = (obs + mu * wsum[:, None, None] * avg) / denom
    return cur
