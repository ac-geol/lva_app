"""Tune lsq_gradient against the logged contact surfaces.

Judged on three numbers that can disagree, and must all be reported:
  contact_err  -- median angle to logged reference surfaces (external truth)
  stability    -- split-half over holes (does the answer survive redrilling?)
  vs_reference -- distance from the pure drill-geometry answer (is it geology?)

Smoothing is the trap: it drives spatial coherence toward zero while leaving
stability flat, so it manufactures the appearance of structure. Only contact
error and stability are trusted here.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import make_cfg
from core.contacts import compare_to_planes, extract_contacts, fit_contact_planes
from core.estimators import field_to_orientations, lsq_gradient, shape_pca
from core.geometry import angular_difference
from core.ingest import load_tables
from core.neighbors import EdgeSet, build_edge_set, build_edges, edge_weights
from core.pipeline import prepare
from core.smoothing import smooth_scores


def stability(coords, scores, lengths, hole_code, cfg, iters, alpha, seed=0):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=int(hole_code.max()) + 1)
    i, j, d, u = build_edges(coords, cfg['radius_m'], hole_code,
                             exclude_same_hole=True)
    fields = []
    for g in (0, 1):
        k = group[hole_code[j]] == g
        ii, jj, dd, uu = i[k], j[k], d[k], u[k]
        ww = edge_weights(ii, jj, dd, scores, lengths, hole_code,
                          distance_sigma_m=cfg['distance_sigma_m'],
                          decluster_by_hole=True)
        e = EdgeSet(i=ii, j=jj, d=dd, u=uu, w=ww, n_nodes=len(coords))
        sc = smooth_scores(scores, e, iterations=iters, alpha=alpha)
        half = dict(cfg)
        half['min_neighbors'] = max(1, cfg['min_neighbors'] // 2)
        half['min_holes'] = max(1, cfg.get('min_holes', 0) // 2)
        fields.append(field_to_orientations(
            lsq_gradient(e, coords, sc, hole_code, half)))
    a, b = fields
    both = a['valid'].to_numpy() & b['valid'].to_numpy()
    if both.sum() == 0:
        return np.nan
    cols = ['pole_vec_x', 'pole_vec_y', 'pole_vec_z']
    return float(np.median(angular_difference(a.loc[both, cols].to_numpy(),
                                              b.loc[both, cols].to_numpy())))


# Bring your own tables. `data/` is gitignored for exactly this; edit these
# paths to match your filenames.
SAMPLES = 'data/samples.csv'
COLLARS = 'data/collars.csv'
SURVEYS = 'data/surveys.csv'
INTERP = 'data/logged_intervals.csv'


def main():
    cfg0 = make_cfg()
    s, c, v = load_tables(SAMPLES, COLLARS, SURVEYS, cfg0)
    contacts = extract_contacts(pd.read_csv(INTERP), c, v)
    planes = fit_contact_planes(contacts, min_holes=8)
    ds = prepare(s, c, v, cfg0)
    xyz = ds.active[['mid_x', 'mid_y', 'mid_z']].reset_index(drop=True)

    rows = []
    for radius, (iters, alpha) in itertools.product(
            [60.0, 90.0, 120.0, 180.0],
            [(0, 0.0), (1, 0.5), (2, 0.5), (3, 0.7), (5, 0.7)]):
        cfg = make_cfg(radius_m=radius,
                       min_neighbors=max(15, int(100 * (radius / 120.0) ** 2)),
                       min_holes=3, distance_sigma_m=radius * 0.67)
        edges = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code, cfg)
        sc = smooth_scores(ds.scores, edges, iterations=iters, alpha=alpha)

        out = field_to_orientations(lsq_gradient(edges, ds.coords, sc,
                                                 ds.hole_code, cfg))
        ctrl = field_to_orientations(shape_pca(
            build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code, cfg,
                           use_score_weight=False),
            ds.coords, ds.scores, ds.hole_code, cfg))
        sel = out['valid'].to_numpy() & ctrl['valid'].to_numpy()
        if sel.sum() < 300:
            continue

        cols = ['pole_vec_x', 'pole_vec_y', 'pole_vec_z']
        vs_ctrl = float(np.median(angular_difference(
            out.loc[sel, cols].to_numpy(), ctrl.loc[sel, cols].to_numpy())))

        joined = pd.concat([xyz, out], axis=1).loc[out['valid']].reset_index(drop=True)
        cmp = compare_to_planes(joined, planes, contacts)
        if cmp.empty:
            continue
        err = float(np.average(cmp['median_err_deg'], weights=cmp['n_nodes']))

        rows.append({'radius': radius, 'min_n': cfg['min_neighbors'],
                     'smooth': f"{iters}x{alpha}", 'n_valid': int(out['valid'].sum()),
                     'contact_err': err,
                     'stability': stability(ds.coords, ds.scores, ds.lengths,
                                            ds.hole_code, cfg, iters, alpha),
                     'vs_reference': vs_ctrl,
                     'r2_med': float(out.loc[out['valid'], 'gradient_r2'].median()),
                     'cond_med': float(out.loc[out['valid'],
                                               'sampling_conditioning'].median())})

    df = pd.DataFrame(rows).sort_values('contact_err')
    Path('out').mkdir(exist_ok=True)
    df.to_csv('out/lsq_tuning.csv', index=False)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))


if __name__ == '__main__':
    main()
