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

    # ---- support ----
    # Downhole composite length in metres. None => use the assay intervals as
    # supplied (MacPass: ~1.3 m mean). Compositing runs on raw grades before
    # scoring and before the neighbourhood graph exists -- see core/compositing.
    'composite_length_m': None,
    # Gaps wider than this split a run instead of sitting inside a composite.
    # None => composite_length_m.
    'composite_gap_tolerance_m': None,
    # A run's final cell shorter than this fraction of the composite length is
    # merged into the preceding composite.
    'composite_min_tail_fraction': 0.5,
    # Drop composites whose assayed length falls below this fraction of the
    # composite length. 0.0 disables the filter (honest default until calibrated).
    'composite_min_coverage': 0.0,

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
    # 'shape_pca' | 'structure_tensor' | 'edge_tensor' | 'lsq_gradient'
    'estimator': 'lsq_gradient',

    # structure_tensor: drop nodes whose neighbourhood is too flat to orient.
    # 0.0 disables the filter (the honest default until calibrated per dataset).
    'min_gradient_energy': 0.0,

    # edge_tensor: score difference at which an edge is downweighted by 1/sqrt(e).
    # None => 0.5 * std(active scores).
    'grade_sim_sigma': None,

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
