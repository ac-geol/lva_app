"""Reference surfaces from logged geology -- the closest thing to ground truth
available when there is no oriented core.

A named stratigraphic unit picked in many holes defines a surface: take its top
(or base) contact once per hole, desurvey it, and fit a plane. Because only ONE
point per hole enters each fit, downhole collinearity -- the bias that ruins
shape-PCA on sample midpoints -- cannot enter here.

It is not immune to *fence* bias: if every contributing collar sits on one
section line the fit is unconstrained about that line. `collar_aspect` reports
exactly that, and fits below the threshold must be discarded rather than
trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .desurvey import build_hole_path
from .geometry import normal_to_strike_dip, unitize
from .schema import COLS


def extract_contacts(interp: pd.DataFrame, collars: pd.DataFrame,
                     surveys: pd.DataFrame, *, hole_col='holeid',
                     from_col='from', to_col='to', code_col='Code'
                     ) -> pd.DataFrame:
    """One top and one base contact per (hole, code), desurveyed to XYZ."""
    iv = interp.rename(columns={hole_col: COLS['hole'], from_col: 'from_m',
                                to_col: 'to_m', code_col: 'code'})
    iv = iv[[COLS['hole'], 'from_m', 'to_m', 'code']].dropna()
    iv[COLS['hole']] = iv[COLS['hole']].astype(str)

    picks = (iv.groupby([COLS['hole'], 'code'])
               .agg(top_m=('from_m', 'min'), base_m=('to_m', 'max'))
               .reset_index())

    collar_map = collars.copy()
    collar_map[COLS['hole']] = collar_map[COLS['hole']].astype(str)
    collar_map = collar_map.set_index(COLS['hole'])
    survey_groups = {str(k): v for k, v in
                     surveys.groupby(surveys[COLS['hole']].astype(str), sort=False)}

    rows = []
    for hole_id, g in picks.groupby(COLS['hole'], sort=False):
        if hole_id not in collar_map.index or hole_id not in survey_groups:
            continue
        depths = np.unique(np.concatenate([g['top_m'].values, g['base_m'].values]))
        path = build_hole_path(collar_map.loc[hole_id], survey_groups[hole_id],
                               depths)
        lookup = path.set_index(COLS['depth'])[['X', 'Y', 'Z']]
        for _, r in g.iterrows():
            for kind, depth in (('top', r['top_m']), ('base', r['base_m'])):
                xyz = lookup.loc[depth]
                rows.append({COLS['hole']: hole_id, 'code': r['code'],
                             'boundary': kind, 'depth_m': depth,
                             'x': xyz['X'], 'y': xyz['Y'], 'z': xyz['Z']})
    return pd.DataFrame(rows)


def fit_contact_planes(contacts: pd.DataFrame, *, min_holes: int = 8,
                       min_collar_aspect: float = 0.15) -> pd.DataFrame:
    """Least-squares plane through each surface's contact points.

    collar_aspect is the ratio of the minor to major horizontal spread of the
    contributing points. Near zero means a single drill fence, which cannot
    constrain a plane -- those fits are marked unusable.
    """
    rows = []
    for (code, boundary), g in contacts.groupby(['code', 'boundary']):
        g = g.drop_duplicates(subset=[COLS['hole']])
        if len(g) < min_holes:
            continue
        pts = g[['x', 'y', 'z']].to_numpy()
        centred = pts - pts.mean(axis=0)

        evals, evecs = np.linalg.eigh(centred.T @ centred / len(pts))
        pole = unitize(evecs[:, 0])                    # smallest variance
        strike, dip = normal_to_strike_dip(pole)

        h = np.linalg.eigvalsh(centred[:, :2].T @ centred[:, :2] / len(pts))
        aspect = float(np.sqrt(max(h[0], 0) / h[1])) if h[1] > 0 else 0.0
        resid = centred @ pole

        rows.append({
            'code': code, 'boundary': boundary, 'n_holes': len(g),
            'strike_deg': float(strike), 'dip_deg': float(dip),
            'pole_x': pole[0], 'pole_y': pole[1], 'pole_z': pole[2],
            'rms_resid_m': float(np.sqrt((resid ** 2).mean())),
            'planarity': float(1.0 - evals[0] / evals[2]) if evals[2] > 0 else np.nan,
            'collar_aspect': aspect,
            'extent_m': float(np.linalg.norm(pts.max(0) - pts.min(0))),
            'usable': bool(aspect >= min_collar_aspect),
            'cx': pts[:, 0].mean(), 'cy': pts[:, 1].mean(), 'cz': pts[:, 2].mean(),
        })
    return pd.DataFrame(rows).sort_values('n_holes', ascending=False)


def compare_to_planes(orientations: pd.DataFrame, planes: pd.DataFrame,
                      contacts: pd.DataFrame, *, max_dist_m: float = 100.0
                      ) -> pd.DataFrame:
    """Median angle between estimated poles and each reference surface's pole.

    Only nodes lying within `max_dist_m` of an actual contact point for that
    surface are compared, so a surface is judged where it was mapped.
    """
    from scipy.spatial import cKDTree

    from .geometry import angular_difference

    node_xyz = orientations[['mid_x', 'mid_y', 'mid_z']].to_numpy()
    node_poles = orientations[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
    tree = cKDTree(node_xyz)

    rows = []
    for _, p in planes.loc[planes['usable']].iterrows():
        pts = contacts.loc[(contacts['code'] == p['code'])
                           & (contacts['boundary'] == p['boundary']),
                           ['x', 'y', 'z']].to_numpy()
        near = sorted({k for grp in tree.query_ball_point(pts, r=max_dist_m)
                       for k in grp})
        if len(near) < 20:
            continue
        ref = np.array([p['pole_x'], p['pole_y'], p['pole_z']])
        ang = angular_difference(node_poles[near],
                                 np.broadcast_to(ref, (len(near), 3)))
        rows.append({'code': p['code'], 'boundary': p['boundary'],
                     'n_holes': int(p['n_holes']), 'n_nodes': len(near),
                     'ref_strike': p['strike_deg'], 'ref_dip': p['dip_deg'],
                     'median_err_deg': float(np.median(ang)),
                     'p25_err_deg': float(np.percentile(ang, 25))})
    return pd.DataFrame(rows).sort_values('n_nodes', ascending=False)
