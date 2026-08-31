"""Can the structure tensor be rescued? Sweep scale, score smoothing, and
tensor smoothing, judged on split-half stability rather than on how tight the
stereonet looks.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import make_cfg
from core.estimators import apply_smoothing, field_to_orientations, structure_tensor
from core.geometry import angular_difference
from core.ingest import load_tables
from core.neighbors import EdgeSet, build_edge_set, build_edges, edge_weights
from core.pipeline import prepare
from core.smoothing import smooth_scores
from core.validation import coherence_null, neighbor_agreement


def stability(coords, scores, lengths, hole_code, cfg, smooth_iters, alpha, mu,
              seed=0):
    """Split-half stability with the full smoothing chain applied to each half."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=int(hole_code.max()) + 1)
    i, j, d, u = build_edges(coords, cfg['radius_m'], hole_code,
                             exclude_same_hole=True)
    fields = []
    for g in (0, 1):
        keep = group[hole_code[j]] == g
        ii, jj, dd, uu = i[keep], j[keep], d[keep], u[keep]
        ww = edge_weights(ii, jj, dd, scores, lengths, hole_code,
                          distance_sigma_m=cfg['distance_sigma_m'],
                          decluster_by_hole=True)
        e = EdgeSet(i=ii, j=jj, d=dd, u=uu, w=ww, n_nodes=len(coords))
        s = smooth_scores(scores, e, iterations=smooth_iters, alpha=alpha)
        half = dict(cfg)
        half['min_neighbors'] = max(1, cfg['min_neighbors'] // 2)
        half['min_holes'] = max(1, cfg.get('min_holes', 0) // 2)
        tf = apply_smoothing(structure_tensor(e, coords, s, hole_code, half),
                             e, mu=mu)
        fields.append(field_to_orientations(tf))
    a, b = fields
    both = a['valid'].to_numpy() & b['valid'].to_numpy()
    if both.sum() == 0:
        return np.nan, 0
    cols = ['pole_vec_x', 'pole_vec_y', 'pole_vec_z']
    return float(np.median(angular_difference(a.loc[both, cols].to_numpy(),
                                              b.loc[both, cols].to_numpy()))), int(both.sum())


def main():
    cfg0 = make_cfg()
    s, c, v = load_tables('MPA_Samples_BD_20240227.csv',
                          'MPA_Collar_20240227.csv',
                          'MPA_Survey_20240227.csv', cfg0)
    ds = prepare(s, c, v, cfg0)
    print(f"active nodes: {len(ds.active):,}\n")

    rows = []
    radii = [40.0, 60.0, 80.0, 120.0, 180.0]
    smooth_opts = [(0, 0.0), (1, 0.5), (2, 0.5), (3, 0.7)]
    mus = [0.0, 1.0, 5.0]

    for radius, (iters, alpha), mu in itertools.product(radii, smooth_opts, mus):
        cfg = make_cfg(radius_m=radius,
                       min_neighbors=max(10, int(100 * (radius / 120.0) ** 2)),
                       min_holes=3, distance_sigma_m=radius * 0.67)
        edges = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code, cfg)
        sc = smooth_scores(ds.scores, edges, iterations=iters, alpha=alpha)
        tf = apply_smoothing(
            structure_tensor(edges, ds.coords, sc, ds.hole_code, cfg),
            edges, mu=mu)
        out = field_to_orientations(tf)
        out = out.loc[out['valid']]
        if len(out) < 200:
            continue

        nodes = np.flatnonzero(tf.valid)
        poles = out[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
        coh = float(np.nanmedian(neighbor_agreement(poles, edges, nodes)))
        null = coherence_null(poles, edges, nodes)
        stab, n_stab = stability(ds.coords, ds.scores, ds.lengths, ds.hole_code,
                                 cfg, iters, alpha, mu)

        rows.append({'radius': radius, 'min_n': cfg['min_neighbors'],
                     'smooth_iters': iters, 'alpha': alpha, 'mu': mu,
                     'n_valid': len(out),
                     'planarity_norm': out['planarity_norm'].median(),
                     'coherence': coh, 'null': null, 'stability': stab,
                     'gain_vs_null': null - coh})

    df = pd.DataFrame(rows).sort_values('stability')
    Path('out').mkdir(exist_ok=True)
    df.to_csv('out/structure_tensor_sweep.csv', index=False)
    pd.set_option('display.width', 200)
    print("BEST 15 by split-half stability (lower = better; ~60 = noise)")
    print(df.head(15).to_string(index=False,
                                float_format=lambda x: f"{x:7.2f}"))
    print("\nWORST 5")
    print(df.tail(5).to_string(index=False, float_format=lambda x: f"{x:7.2f}"))


if __name__ == '__main__':
    main()
