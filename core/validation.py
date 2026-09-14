"""Ways to judge an orientation field when there is no measured structure.

Without oriented core the only honest questions are internal ones:

  1. SPATIAL COHERENCE -- do nearby nodes, in *different* holes, agree? A real
     geological field varies smoothly. Noise does not. The null is 60 degrees:
     the median acute angle between two random axes.

  2. SPLIT-HALF STABILITY -- assign every hole to group A or B, estimate each
     node twice using only A-holes then only B-holes, and compare. If the
     answer depends on which holes happened to be drilled, it is a property of
     the drill programme, not of the rock. This is the sharpest test available
     without ground truth.

Both return angles in degrees, where lower is better and ~60 means no signal.
"""
from __future__ import annotations

import numpy as np

from .geometry import angular_difference
from .neighbors import EdgeSet, build_edges, edge_weights


def neighbor_agreement(poles: np.ndarray, edges: EdgeSet,
                       node_ids: np.ndarray | None = None) -> np.ndarray:
    """Median angle between each node's pole and its neighbours' poles."""
    n = edges.n_nodes
    if node_ids is None:
        node_ids = np.arange(n)

    slot = np.full(n, -1, dtype=np.intp)
    slot[node_ids] = np.arange(len(node_ids))
    ok = (slot[edges.i] >= 0) & (slot[edges.j] >= 0)
    i, j = edges.i[ok], edges.j[ok]
    if len(i) == 0:
        return np.array([])

    ang = angular_difference(poles[slot[i]], poles[slot[j]])

    order = np.argsort(i, kind='stable')
    i_s, ang_s = i[order], ang[order]
    bounds = np.searchsorted(i_s, np.arange(n + 1))
    out = np.full(n, np.nan)
    for node in range(n):
        lo, hi = bounds[node], bounds[node + 1]
        if hi > lo:
            out[node] = np.median(ang_s[lo:hi])
    return out[node_ids]


def coherence_null(poles: np.ndarray, edges: EdgeSet, node_ids=None,
                   seed: int = 0) -> float:
    """Same statistic after shuffling poles between nodes -- the no-signal case."""
    rng = np.random.default_rng(seed)
    shuffled = poles[rng.permutation(len(poles))]
    return float(np.nanmedian(neighbor_agreement(shuffled, edges, node_ids)))


def split_half_stability(estimator, coords, scores, lengths, hole_code, cfg,
                         *, seed: int = 0, use_score_weight: bool = True):
    """Estimate every node twice from disjoint halves of the holes.

    Returns (median_angle_deg, n_compared). The node itself is never dropped --
    only which *other* holes are allowed to inform it -- so this isolates
    sensitivity to the drill programme rather than to sample count.
    """
    from .estimators import field_to_orientations

    rng = np.random.default_rng(seed)
    n_holes = int(hole_code.max()) + 1
    group = rng.integers(0, 2, size=n_holes)

    i, j, d, u = build_edges(
        coords, cfg['radius_m'], hole_code,
        exclude_same_hole=cfg.get('exclude_same_hole_neighbors', True))

    fields = []
    for g in (0, 1):
        keep = group[hole_code[j]] == g
        ii, jj, dd, uu = i[keep], j[keep], d[keep], u[keep]
        ww = edge_weights(ii, jj, dd, scores, lengths, hole_code,
                          distance_sigma_m=cfg['distance_sigma_m'],
                          decluster_by_hole=cfg.get('decluster_by_hole', True),
                          use_score_weight=use_score_weight)
        edges = EdgeSet(i=ii, j=jj, d=dd, u=uu, w=ww, n_nodes=len(coords))

        # Halving the holes halves the neighbours; relax the threshold to match
        # so this measures instability, not a different validity filter.
        half_cfg = dict(cfg)
        half_cfg['min_neighbors'] = max(1, cfg['min_neighbors'] // 2)
        half_cfg['min_holes'] = max(1, cfg.get('min_holes', 0) // 2)

        tf = estimator(edges, coords, scores, hole_code, half_cfg)
        fields.append(field_to_orientations(tf))

    a, b = fields
    both = a['valid'].to_numpy() & b['valid'].to_numpy()
    if both.sum() == 0:
        return float('nan'), 0

    cols = ['pole_vec_x', 'pole_vec_y', 'pole_vec_z']
    ang = angular_difference(a.loc[both, cols].to_numpy(),
                             b.loc[both, cols].to_numpy())
    return float(np.median(ang)), int(both.sum())
