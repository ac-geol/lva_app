"""Default configuration. One dict, overridden per run."""
from __future__ import annotations

import copy

DEFAULT_CFG = {
    # ---- data filters (applied to collar metadata after standardization) ----
    'property_filter': None,
    'prospect_filter': None,
    'hole_filter': None,

    # ---- data quality ----
    'exclude_low_recovery': False,
    'low_recovery_col': 'LowRecovery_<=85pct',

    # ---- scoring ----
    'score_columns': ['Ag_ppm', 'Pb_pct', 'Zn_pct'],
    'log_transform_score': True,
    'active_score_quantile': 0.85,

    # ---- neighbourhood ----
    'radius_m': 120.0,
    'min_neighbors': 100,
    'distance_sigma_m': 80.0,
    'exclude_same_hole_neighbors': True,
    # Minimum distinct contributing holes. With same-hole edges excluded this
    # is the honest sample size; a node fed by two holes cannot resolve a plane.
    'min_holes': 3,

    # Equalize each contributing hole's influence within a neighbourhood, so a
    # densely sampled hole does not outvote a sparsely sampled one.
    'decluster_by_hole': True,

    # ---- estimator ----
    # 'shape_pca' (default) | 'lsq_gradient'
    # shape_pca ships as the default on the evidence available: it wins against
    # synthetic ground truth (15.6 deg vs 21.9) and against logged contacts
    # (21.5 vs 29.2). CLAUDE.md section 5 records the one experiment that
    # could overturn that -- a single-azimuth drill programme, still untried.
    'estimator': 'shape_pca',

    # lsq_gradient: ridge on the normal equations, relative to each node's own
    # tensor magnitude. Large enough to survive a degenerate drill pattern,
    # small enough not to touch a well-sampled one.
    'lsq_ridge': 1e-3,
    # Drop nodes whose sampling-direction tensor is too degenerate to constrain
    # a gradient (lambda_min/lambda_max). 0.0 disables the filter.
    'min_sampling_conditioning': 0.0,

    # ---- column mapping (None => auto-detect from aliases) ----
    'columns': {
        'samples':  {'hole': None, 'from': None, 'to': None},
        # One table of already-desurveyed points, as Leapfrog and Datamine
        # export it. Coordinates live on the rows, so no collar or survey
        # table is involved. See core/ingest.load_points.
        'points':   {'hole': None, 'from': None, 'to': None,
                     'easting': None, 'northing': None, 'elevation': None},
        'collars':  {'hole': None, 'easting': None, 'northing': None,
                     'elevation': None, 'length': None, 'dip': None,
                     'azimuth': None, 'property': None, 'prospect': None,
                     'holetype': None},
        'surveys':  {'hole': None, 'depth': None, 'dip': None, 'azimuth': None},
    },
    'auto_detect_columns': True,
}


def make_cfg(**overrides) -> dict:
    """Deep copy of the defaults with top-level overrides applied."""
    cfg = copy.deepcopy(DEFAULT_CFG)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    cfg.update(overrides)
    return cfg
