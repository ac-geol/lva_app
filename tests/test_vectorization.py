"""The vectorized engine must reproduce the notebook's per-point loop exactly.

The notebook computed each point's orientation inside a Python for-loop. That
loop is gone -- replaced by six bincounts. These tests pin the replacement to
the original arithmetic so the rewrite cannot have changed any result.
"""
import numpy as np
import pytest

from core.config import make_cfg
from core.estimators import shape_pca, structure_tensor, field_to_orientations
from core.geometry import angular_difference
from core.neighbors import EdgeSet, build_edge_set, build_edges, edge_weights


def _synthetic(n=600, seed=5):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 300, size=(n, 3))
    scores = rng.normal(size=n)
    lengths = rng.uniform(0.5, 3.0, size=n)
    hole_code = rng.integers(0, 25, size=n).astype(np.int64)
    return coords, scores, lengths, hole_code


def _weighted_pca_reference(local_coords, weights):
    """Verbatim transcription of the notebook's weighted_pca()."""
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1e-6)
    weights = weights / weights.sum()
    centroid = np.sum(local_coords * weights[:, None], axis=0)
    xc = local_coords - centroid
    cov = (xc * weights[:, None]).T @ xc
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return centroid, evals[order], evecs[:, order]


def test_shape_pca_matches_the_per_point_loop():
    coords, scores, lengths, hole_code = _synthetic()
    cfg = make_cfg(radius_m=80.0, min_neighbors=10, min_holes=3,
                   distance_sigma_m=60.0)
    edges = build_edge_set(coords, scores, lengths, hole_code, cfg)
    tf = shape_pca(edges, coords, scores, hole_code, cfg)

    # Reference: loop over nodes, rebuild each neighbourhood, run the original PCA.
    order = np.argsort(edges.i, kind='stable')
    i_s, j_s, w_s = edges.i[order], edges.j[order], edges.w[order]
    bounds = np.searchsorted(i_s, np.arange(edges.n_nodes + 1))

    checked = 0
    for node in range(edges.n_nodes):
        lo, hi = bounds[node], bounds[node + 1]
        if hi - lo < 10:
            continue
        _, evals, evecs = _weighted_pca_reference(coords[j_s[lo:hi]], w_s[lo:hi])
        got = np.linalg.eigvalsh(tf.T[node])[::-1]
        assert got == pytest.approx(evals, rel=1e-9, abs=1e-12)
        checked += 1
    assert checked > 100


def test_structure_tensor_matches_a_naive_loop():
    coords, scores, lengths, hole_code = _synthetic()
    cfg = make_cfg(radius_m=80.0, min_neighbors=10, min_holes=3,
                   distance_sigma_m=60.0)
    edges = build_edge_set(coords, scores, lengths, hole_code, cfg)
    tf = structure_tensor(edges, coords, scores, hole_code, cfg)

    ref = np.zeros_like(tf.T)
    wsum = np.zeros(edges.n_nodes)
    for e in range(edges.n_edges):
        i, j = edges.i[e], edges.j[e]
        delta = (scores[j] - scores[i]) / edges.d[e]
        u = edges.u[e]
        ref[i] += edges.w[e] * delta ** 2 * np.outer(u, u)
        wsum[i] += edges.w[e]
    ref /= np.where(wsum > 0, wsum, 1.0)[:, None, None]

    assert tf.T == pytest.approx(ref, rel=1e-9, abs=1e-14)


def test_edges_exclude_self_and_same_hole():
    coords, scores, lengths, hole_code = _synthetic()
    i, j, d, u = build_edges(coords, 80.0, hole_code, exclude_same_hole=True)
    assert not np.any(i == j)
    assert not np.any(hole_code[i] == hole_code[j])
    assert np.all(d > 0)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)


def test_same_hole_edges_appear_when_not_excluded():
    coords, scores, lengths, hole_code = _synthetic()
    i, j, _, _ = build_edges(coords, 80.0, hole_code, exclude_same_hole=False)
    assert np.any(hole_code[i] == hole_code[j])


def test_declustering_equalizes_hole_influence():
    """A hole with many samples must not outvote one with few."""
    rng = np.random.default_rng(9)
    # Node 0 at the origin; hole 1 contributes 50 samples, hole 2 contributes 2,
    # all at comparable distance.
    pts = [np.zeros(3)]
    codes = [0]
    for _ in range(50):
        pts.append(rng.normal(scale=5.0, size=3) + np.array([40.0, 0, 0]))
        codes.append(1)
    for _ in range(2):
        pts.append(rng.normal(scale=5.0, size=3) + np.array([0.0, 40.0, 0]))
        codes.append(2)
    coords = np.array(pts)
    hole_code = np.array(codes, dtype=np.int64)
    scores = np.ones(len(coords))
    lengths = np.ones(len(coords))

    i, j, d, u = build_edges(coords, 80.0, hole_code, exclude_same_hole=True)
    raw = edge_weights(i, j, d, scores, lengths, hole_code,
                       distance_sigma_m=60.0, decluster_by_hole=False)
    dec = edge_weights(i, j, d, scores, lengths, hole_code,
                       distance_sigma_m=60.0, decluster_by_hole=True)

    node0 = i == 0
    h1 = node0 & (hole_code[j] == 1)
    h2 = node0 & (hole_code[j] == 2)

    # Undeclustered, the 50-sample hole dominates by an order of magnitude.
    assert raw[h1].sum() / raw[h2].sum() > 10
    # Declustered, the two holes carry comparable total weight.
    assert 0.3 < dec[h1].sum() / dec[h2].sum() < 3.0
