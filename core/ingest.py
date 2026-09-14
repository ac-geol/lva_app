"""Load and clean drillhole data, by either of two routes.

`load_tables` takes the three raw tables (samples, collars, surveys) and leaves
desurveying to the pipeline. `load_points` takes one table of already-desurveyed
points, as exported by Leapfrog or Datamine, and needs no collar or survey data.

Both accept paths or file-like objects, so the same code serves a CLI, a
notebook, or a browser upload.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COLS, REQUIRED, rename_to_internal

_NA_MARKERS = ['-', '', 'NA', 'N/A', 'null', 'None']


def _read(src) -> pd.DataFrame:
    df = src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    return df.replace(_NA_MARKERS, np.nan)


def load_tables(samples_src, collars_src, surveys_src, cfg: dict):
    """Return (samples, collars, surveys) with internal names and clean types."""
    auto = cfg.get('auto_detect_columns', True)
    colmap = cfg.get('columns', {})

    samples = rename_to_internal(_read(samples_src), colmap.get('samples'),
                                 REQUIRED['samples'], auto_detect=auto)
    collars = rename_to_internal(_read(collars_src), colmap.get('collars'),
                                 REQUIRED['collars'], auto_detect=auto)
    surveys = rename_to_internal(_read(surveys_src), colmap.get('surveys'),
                                 REQUIRED['surveys'], auto_detect=auto)

    for c in [COLS['from'], COLS['to']] + list(cfg.get('score_columns', [])):
        if c in samples.columns:
            samples[c] = pd.to_numeric(samples[c], errors='coerce')
    for c in [COLS['easting'], COLS['northing'], COLS['elev'],
              COLS['length'], COLS['dip'], COLS['azm']]:
        if c in collars.columns:
            collars[c] = pd.to_numeric(collars[c], errors='coerce')
    for c in [COLS['depth'], COLS['dip'], COLS['azm']]:
        if c in surveys.columns:
            surveys[c] = pd.to_numeric(surveys[c], errors='coerce')

    samples['interval_m'] = samples[COLS['to']] - samples[COLS['from']]
    samples['mid_m'] = 0.5 * (samples[COLS['from']] + samples[COLS['to']])
    samples = samples.loc[samples['interval_m'] > 0].copy()

    if cfg.get('exclude_low_recovery') and cfg['low_recovery_col'] in samples.columns:
        flag = samples[cfg['low_recovery_col']].fillna('N').astype(str).str.upper()
        samples = samples.loc[flag != 'Y'].copy()

    # A hole is only usable if it has a collar and at least one survey station.
    valid = (set(collars[COLS['hole']].astype(str))
             & set(surveys[COLS['hole']].astype(str)))
    samples, collars, surveys = (
        _keep_holes(t, valid) for t in (samples, collars, surveys))

    samples, collars, surveys = apply_filters(samples, collars, surveys, collars, cfg)
    return samples, collars, surveys


def _keep_holes(df: pd.DataFrame, holes: set) -> pd.DataFrame:
    return df.loc[df[COLS['hole']].astype(str).isin(holes)].copy()


def select_holes(collar_meta: pd.DataFrame, cfg: dict) -> set | None:
    """Hole IDs surviving the Property / Prospect / explicit-hole filters.

    None means "no filter configured", which is distinct from an empty set.
    Split out of apply_filters so the single-table point path shares exactly
    these semantics rather than reimplementing them.
    """
    hole = COLS['hole']
    keep = None

    for cfg_key, col in [('property_filter', COLS['property']),
                         ('prospect_filter', COLS['prospect'])]:
        want = cfg.get(cfg_key)
        if want is not None and col in collar_meta.columns:
            sel = set(collar_meta.loc[
                collar_meta[col].astype(str) == str(want), hole].astype(str))
            keep = sel if keep is None else (keep & sel)

    if cfg.get('hole_filter') is not None:
        sel = set(map(str, cfg['hole_filter']))
        keep = sel if keep is None else (keep & sel)

    return keep


def apply_filters(samples, collars, surveys, collar_meta, cfg: dict):
    """Restrict to a Property / Prospect / explicit hole list, if configured."""
    keep = select_holes(collar_meta, cfg)
    if keep is None:
        return samples, collars, surveys
    return tuple(_keep_holes(t, keep) for t in (samples, collars, surveys))


def load_points(points_src, cfg: dict) -> pd.DataFrame:
    """Load one table of already-desurveyed sample points.

    This is the Leapfrog / Datamine export path: the coordinates are in the
    file, so there is no collar table, no survey table and no desurveying. The
    returned frame carries the same internal names and derived columns as
    `load_tables` produces for samples, plus mid_x / mid_y / mid_z taken
    directly from the file's coordinates.

    Coordinates are read as the location of the interval MIDPOINT, which is
    what both packages export for a desurveyed interval table.
    """
    auto = cfg.get('auto_detect_columns', True)
    colmap = cfg.get('columns', {})

    pts = rename_to_internal(_read(points_src), colmap.get('points'),
                             REQUIRED['points'], auto_detect=auto)

    numeric = ([COLS['from'], COLS['to'], COLS['easting'], COLS['northing'],
                COLS['elev']] + list(cfg.get('score_columns', [])))
    for c in numeric:
        if c in pts.columns:
            pts[c] = pd.to_numeric(pts[c], errors='coerce')

    pts['interval_m'] = pts[COLS['to']] - pts[COLS['from']]
    pts['mid_m'] = 0.5 * (pts[COLS['from']] + pts[COLS['to']])
    pts = pts.loc[pts['interval_m'] > 0].copy()

    if cfg.get('exclude_low_recovery') and cfg['low_recovery_col'] in pts.columns:
        flag = pts[cfg['low_recovery_col']].fillna('N').astype(str).str.upper()
        pts = pts.loc[flag != 'Y'].copy()

    # A point with no coordinate cannot be a node, and carrying it forward only
    # produces a NaN that survives into the neighbourhood graph.
    coords = [COLS['easting'], COLS['northing'], COLS['elev']]
    pts = pts.dropna(subset=coords).copy()

    # The point table is its own collar metadata: Property / Prospect travel on
    # the rows when the exporting package was asked to include them.
    keep = select_holes(pts, cfg)
    if keep is not None:
        pts = _keep_holes(pts, keep)

    return pts.reset_index(drop=True)
