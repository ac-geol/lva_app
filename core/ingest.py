"""Load and clean the three drillhole tables.

Accepts paths or file-like objects, so the same code serves a CLI, a notebook,
or a browser upload.
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


def apply_filters(samples, collars, surveys, collar_meta, cfg: dict):
    """Restrict to a Property / Prospect / explicit hole list, if configured."""
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

    if keep is None:
        return samples, collars, surveys
    return tuple(_keep_holes(t, keep) for t in (samples, collars, surveys))
