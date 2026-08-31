"""Desurvey: turn (hole, depth) into (X, Y, Z) by minimum-curvature integration.

Coordinates are East / North / Up. Dip is negative downward, matching the
convention in the collar and survey tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COLS


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def dip_az_to_vector(dip_deg, az_deg) -> np.ndarray:
    """dip/azimuth (degrees) -> unit vectors in East, North, Up."""
    dip = np.radians(np.asarray(dip_deg, dtype=float))
    az = np.radians(np.asarray(az_deg, dtype=float))
    return normalize_rows(np.column_stack([
        np.cos(dip) * np.sin(az),
        np.cos(dip) * np.cos(az),
        np.sin(dip),
    ]))


def build_hole_path(collar_row: pd.Series, survey_hole: pd.DataFrame,
                    target_depths: np.ndarray) -> pd.DataFrame:
    """Integrate one hole's trace, evaluated at every survey and target depth."""
    sv = (survey_hole[[COLS['depth'], COLS['dip'], COLS['azm']]]
          .dropna().sort_values(COLS['depth']).copy())

    # Surveys often start below the collar; seed direction from the collar.
    if sv.empty or float(sv[COLS['depth']].iloc[0]) > 0.0:
        seed = pd.DataFrame({COLS['depth']: [0.0],
                             COLS['dip']: [float(collar_row[COLS['dip']])],
                             COLS['azm']: [float(collar_row[COLS['azm']])]})
        sv = pd.concat([seed, sv], ignore_index=True)

    sv = (sv.drop_duplicates(subset=COLS['depth'])
            .sort_values(COLS['depth']).reset_index(drop=True))

    # Extend the last station's attitude past the deepest sample if needed.
    max_depth = max(float(np.nanmax(target_depths)), float(sv[COLS['depth']].max()))
    if max_depth > float(sv[COLS['depth']].max()):
        last = sv.iloc[[-1]].copy()
        last[COLS['depth']] = max_depth
        sv = pd.concat([sv, last], ignore_index=True)

    survey_dirs = dip_az_to_vector(sv[COLS['dip']].values, sv[COLS['azm']].values)
    survey_md = sv[COLS['depth']].values.astype(float)

    depths = np.unique(np.concatenate(
        [survey_md, np.asarray(target_depths, dtype=float)]))

    dirs = np.empty((len(depths), 3))
    for j in range(3):
        dirs[:, j] = np.interp(depths, survey_md, survey_dirs[:, j])
    dirs = normalize_rows(dirs)

    # Minimum curvature: ratio factor rf straightens the chord between stations.
    seg = np.diff(depths)
    u1, u2 = dirs[:-1], dirs[1:]
    dot = np.clip(np.einsum('ij,ij->i', u1, u2), -1.0, 1.0)
    dogleg = np.arccos(dot)
    rf = np.where(dogleg < 1e-12, 1.0,
                  (2.0 / np.where(dogleg < 1e-12, 1.0, dogleg))
                  * np.tan(dogleg / 2.0))
    disp = 0.5 * seg[:, None] * (u1 + u2) * rf[:, None]

    coords = np.empty((len(depths), 3))
    coords[0] = [float(collar_row[COLS['easting']]),
                 float(collar_row[COLS['northing']]),
                 float(collar_row[COLS['elev']])]
    coords[1:] = coords[0] + np.cumsum(disp, axis=0)

    return pd.DataFrame({
        COLS['depth']: depths,
        'X': coords[:, 0], 'Y': coords[:, 1], 'Z': coords[:, 2],
        'ux': dirs[:, 0], 'uy': dirs[:, 1], 'uz': dirs[:, 2],
    })


def desurvey_samples(samples: pd.DataFrame, collars: pd.DataFrame,
                     surveys: pd.DataFrame):
    """Attach from/to/mid XYZ and the downhole direction to every sample.

    Returns (desurveyed_samples, {hole_id: path}).
    """
    hole = COLS['hole']
    collar_map = collars.set_index(hole)
    survey_groups = {str(k): v for k, v in surveys.groupby(
        surveys[hole].astype(str), sort=False)}

    meta_cols = [c for c in (COLS['property'], COLS['prospect'], COLS['holetype'])
                 if c in collars.columns]
    parts, paths = [], {}

    for hole_id, g in samples.groupby(hole, sort=False):
        collar_row = collar_map.loc[hole_id]
        sv = survey_groups.get(str(hole_id), surveys.iloc[0:0])

        targets = np.unique(np.concatenate([
            g[COLS['from']].values.astype(float),
            g[COLS['to']].values.astype(float),
            g['mid_m'].values.astype(float),
        ]))

        path = build_hole_path(collar_row, sv, targets)
        paths[str(hole_id)] = path
        lookup = path.set_index(COLS['depth'])

        gg = g.copy()
        for prefix, depth_col in [('from', COLS['from']), ('to', COLS['to']),
                                  ('mid', 'mid_m')]:
            gg[[f'{prefix}_x', f'{prefix}_y', f'{prefix}_z']] = \
                lookup.loc[gg[depth_col].values, ['X', 'Y', 'Z']].values
        gg[['ux', 'uy', 'uz']] = lookup.loc[gg['mid_m'].values,
                                            ['ux', 'uy', 'uz']].values

        for c in meta_cols:
            gg[c] = collar_row[c]
        gg['hole_length_m'] = collar_row[COLS['length']]
        parts.append(gg)

    des = pd.concat(parts, ignore_index=True)
    return des.reset_index(drop=True).reset_index(names='src_index'), paths
