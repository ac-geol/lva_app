"""Neighbourhood graph construction, shared by every estimator.

Both estimators consume the *same* edge set, so a method comparison compares
estimators rather than accidentally comparing neighbourhoods.

Edges never include self-pairs: a zero-length edge has no direction, and the
structure tensor would divide by its length. For shape-PCA this drops one
point out of a few hundred -- immaterial, and worth it for comparability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class EdgeSet:
    """Directed edges i -> j. Every array is parallel, length n_edges."""
    i: np.ndarray            # source node index
    j: np.ndarray            # neighbour node index
    d: np.ndarray            # separation in metres
    u: np.ndarray            # (n_edges, 3) unit vector from i to j
    w: np.ndarray            # combined weight
    n_nodes: int

    @property
    def n_edges(self) -> int:
        return len(self.i)

    def neighbor_counts(self) -> np.ndarray:
        return np.bincount(self.i, minlength=self.n_nodes)

    def hole_counts(self, hole_code: np.ndarray) -> np.ndarray:
        """Distinct contributing holes per node -- the honest sample size."""
        # The packed key needs int64 range; the bincount argument needs intp.
        # Those are the same type on a 64-bit build and differ under wasm32.
        key = self.i.astype(np.int64) * (hole_code.max() + 1) + hole_code[self.j]
        uniq_key = np.unique(key)
        return np.bincount((uniq_key // (hole_code.max() + 1)).astype(np.intp),
                           minlength=self.n_nodes)


def build_edges(coords: np.ndarray, radius_m: float, hole_code: np.ndarray,
                *, exclude_same_hole: bool = True) -> tuple[np.ndarray, ...]:
    """Radius graph -> (i, j, d, u), self-pairs and optionally same-hole removed."""
    coords = np.ascontiguousarray(coords, dtype=float)
    tree = cKDTree(coords)
    nb = tree.query_ball_point(coords, r=radius_m, return_sorted=False)

    # intp, not int64: this is what indexes arrays, and np.bincount insists on
    # it. They are identical on a 64-bit build; under Pyodide's 32-bit wasm they
    # are not, and an int64 index array is rejected outright.
    lens = np.fromiter((len(x) for x in nb), dtype=np.intp, count=len(nb))
    i = np.repeat(np.arange(len(nb), dtype=np.intp), lens)
    j = np.concatenate([np.asarray(x, dtype=np.intp) for x in nb]) if len(nb) \
        else np.empty(0, dtype=np.intp)

    keep = i != j
    if exclude_same_hole:
        keep &= hole_code[i] != hole_code[j]
    i, j = i[keep], j[keep]

    e = coords[j] - coords[i]
    d = np.linalg.norm(e, axis=1)
    keep = d > 0
    i, j, e, d = i[keep], j[keep], e[keep], d[keep]
    return i, j, d, e / d[:, None]


def _group_ids(i: np.ndarray, hole_code_j: np.ndarray) -> tuple[np.ndarray, int]:
    """One group per (node, contributing hole) pair."""
    key = i.astype(np.int64) * (hole_code_j.max() + 1) + hole_code_j
    _, inv = np.unique(key, return_inverse=True)
    return inv.astype(np.intp), int(inv.max()) + 1 if len(inv) else 0


def _group_max(values: np.ndarray, inv: np.ndarray, n_groups: int) -> np.ndarray:
    order = np.lexsort((values, inv))
    sv, si = values[order], inv[order]
    last = np.flatnonzero(np.r_[np.diff(si) != 0, True])
    out = np.zeros(n_groups)
    out[si[last]] = sv[last]
    return out


def edge_weights(i, j, d, scores, lengths, hole_code, *,
                 distance_sigma_m: float, decluster_by_hole: bool = True,
                 use_score_weight: bool = True) -> np.ndarray:
    """Weight = grade x distance decay x interval length, optionally declustered.

    Declustering normalizes weights within each (node, neighbour-hole) group so
    every contributing hole gets one vote, then rescales the group by its
    closest sample's distance weight -- a hole with 200 samples no longer
    outvotes one with 5, but a nearby hole still outweighs a distant one.
    """
    dist_term = np.exp(-(d ** 2) / (2.0 * distance_sigma_m ** 2))
    len_term = np.clip(lengths[j], 0.1, None)

    if use_score_weight:
        sj = scores[j]
        score_term = np.clip(sj - np.nanmin(scores) + 1e-6, 1e-6, None)
    else:
        score_term = 1.0

    w = score_term * dist_term * len_term

    if decluster_by_hole and len(w):
        inv, n_groups = _group_ids(i, hole_code[j])
        gsum = np.bincount(inv, weights=w, minlength=n_groups)
        gmax_dist = _group_max(dist_term, inv, n_groups)
        w = w / np.where(gsum[inv] > 0, gsum[inv], 1.0) * gmax_dist[inv]

    return w


def build_edge_set(coords, scores, lengths, hole_code, cfg, *,
                   use_score_weight: bool = True) -> EdgeSet:
    """Assemble the shared neighbourhood graph.

    use_score_weight=False strips all grade information from the weights,
    leaving distance and interval length. Running shape-PCA on that is the
    drill-pattern reference: what the drill pattern alone would report.
    """
    i, j, d, u = build_edges(
        coords, cfg['radius_m'], hole_code,
        exclude_same_hole=cfg.get('exclude_same_hole_neighbors', True))
    w = edge_weights(
        i, j, d, scores, lengths, hole_code,
        distance_sigma_m=cfg['distance_sigma_m'],
        decluster_by_hole=cfg.get('decluster_by_hole', True),
        use_score_weight=use_score_weight)
    return EdgeSet(i=i, j=j, d=d, u=u, w=w, n_nodes=len(coords))


def accumulate_tensor(edges: EdgeSet, scalar: np.ndarray,
                      vectors: np.ndarray) -> np.ndarray:
    """Sum_j scalar_ij * v_ij v_ij^T per node, as an (n_nodes, 3, 3) array.

    Six bincounts over the six unique components -- the whole reason the
    orientation loop no longer needs a Python for-loop.
    """
    n = edges.n_nodes
    T = np.zeros((n, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            comp = np.bincount(edges.i,
                               weights=scalar * vectors[:, a] * vectors[:, b],
                               minlength=n)
            T[:, a, b] = comp
            if a != b:
                T[:, b, a] = comp
    return T
