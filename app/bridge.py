"""The browser's entry point into the engine: JSON actions in, results out.

Deliberately outside `core/`, exactly as `scripts/` and `viz/` are. This module
orchestrates -- it chooses the panels, holds the session state between calls,
and packs results for transfer -- but every number it reports is computed by
`core`, unmodified, so the browser and the CLI cannot drift apart.

Protocol
--------
`handle(action, payload_json)` returns a plain dict. Metadata is JSON-ready
Python scalars; bulk geometry is raw `bytes`, which Pyodide hands to JavaScript
as a Uint8Array with no per-element conversion cost. Coordinates are sent as
float32 offsets from a float64 origin -- full UTM values lose millimetres in
float32, and a local origin keeps WebGL free of jitter. The float64 values
never leave Python except through the export, which is written as text.
"""
from __future__ import annotations

import copy
import io
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd

from core.config import DEFAULT_CFG, make_cfg
from core.estimators import ESTIMATORS, field_to_orientations, shape_pca
from core.export import to_leapfrog
from core.geometry import angular_difference
from core.ingest import load_points
from core.neighbors import build_edge_set
from core.pipeline import prepare_points
from core.score import build_mineralization_score
from core.schema import COLS, REQUIRED, describe_mapping, standardize_headers
from core.validation import (coherence_null, neighbor_agreement,
                             split_half_stability)

# Not a method: shape-PCA on weights carrying no grade at all, i.e. what the
# drill pattern alone reports. It runs on every call because the one way the
# default estimator fails is by collapsing onto it.
REFERENCE = 'geometry_control'

# Structural columns are never offered as the assay column.
_STRUCTURAL = {COLS['hole'], COLS['from'], COLS['to'], COLS['easting'],
               COLS['northing'], COLS['elev'], COLS['depth'], COLS['dip'],
               COLS['azm'], COLS['length']}

_STATE: dict = {'csv': None, 'points': None, 'ds': None, 'cfg': None,
                'result': None, 'vs_ref': None}


# --------------------------------------------------------------------------
# Packing helpers.
# --------------------------------------------------------------------------

def _f32(a) -> bytes:
    return np.ascontiguousarray(a, dtype=np.float32).tobytes()


def _i32(a) -> bytes:
    return np.ascontiguousarray(a, dtype=np.int32).tobytes()


def _num(x):
    """A JSON-safe scalar. NaN becomes None so JSON.stringify stays valid."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else x


def _stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors='coerce')
    return {'n': int(s.notna().sum()), 'n_null': int(s.isna().sum()),
            'min': _num(s.min()), 'median': _num(s.median()),
            'max': _num(s.max()), 'mean': _num(s.mean())}


def _cfg_from(payload: dict) -> dict:
    """Build a config from what the UI sent.

    `columns` arrives as {internal_key: source_header} for whatever the user
    corrected by hand. Anything they left alone stays None, which means the
    resolver keeps auto-detecting it.
    """
    overrides = dict(payload.get('cfg') or {})

    column = payload.get('score_column')
    if column:
        overrides['score_columns'] = [column]

    chosen = payload.get('columns') or {}
    if chosen:
        columns = copy.deepcopy(DEFAULT_CFG['columns'])
        columns['points'].update({k: v for k, v in chosen.items() if v})
        overrides['columns'] = columns

    return make_cfg(**overrides)


# --------------------------------------------------------------------------
# Actions.
# --------------------------------------------------------------------------

# The settings the app exposes. Everything else in DEFAULT_CFG stays at its
# default -- a control the geologist cannot act on is just another way to break
# a run.
EXPOSED = ['estimator', 'radius_m', 'min_neighbors', 'min_holes',
           'distance_sigma_m', 'active_score_quantile', 'log_transform_score']


def versions(payload: dict) -> dict:
    """Startup facts: package versions, and the engine's own defaults.

    The defaults are read out of DEFAULT_CFG rather than restated in the page,
    so there is one place a default lives and the UI cannot drift from the
    engine it drives.
    """
    import scipy
    return {'python': sys.version.split()[0], 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__,
            'defaults': {k: DEFAULT_CFG[k] for k in EXPOSED},
            'estimators': sorted(ESTIMATORS)}


def preview(payload: dict) -> dict:
    """Header-only pass: what the resolver would do, and what can be the assay.

    Reads the first rows only. A user correcting a mapping should not wait for
    a 200 MB file to parse before seeing whether the guess was right.
    """
    csv = payload.get('csv') or _STATE['csv']
    _STATE['csv'] = csv

    head = standardize_headers(pd.read_csv(io.StringIO(csv), nrows=2000))
    chosen = payload.get('columns') or {}
    mapping = describe_mapping(head, chosen, REQUIRED['points'])

    # Whatever the mapping claimed is off-limits as the assay: a column cannot
    # be both the hole depth and the grade.
    taken = {row['source'] for _, row in mapping.iterrows() if row['source']}

    numeric = []
    for col in head.columns:
        if col in _STRUCTURAL or col in taken:
            continue
        coerced = pd.to_numeric(head[col], errors='coerce')
        if coerced.notna().mean() > 0.5 and coerced.std(skipna=True) > 0:
            numeric.append({'name': str(col),
                            'median': _num(coerced.median()),
                            'max': _num(coerced.max())})

    records = json.loads(mapping.to_json(orient='records'))
    missing = [r['internal'] for r in records if r['status'] == 'MISSING']

    return {
        'columns': [str(c) for c in head.columns],
        'numeric_columns': numeric,
        'mapping': records,
        'missing': missing,
        'resolved': not missing,
        'n_preview_rows': int(len(head)),
    }


def load(payload: dict) -> dict:
    """Ingest the point table and describe what arrived."""
    csv = payload.get('csv') or _STATE['csv']
    cfg = _cfg_from(payload)
    pts = load_points(io.StringIO(csv), cfg)

    if not len(pts):
        raise ValueError("No usable rows: every interval had zero or negative "
                         "length, or no coordinate.")

    _STATE.update(csv=csv, points=pts, cfg=cfg, ds=None, result=None)

    coords = pts[[COLS['easting'], COLS['northing'], COLS['elev']]].to_numpy(float)
    interval = pts['interval_m']
    median_interval = float(interval.median())

    return {
        'n_rows': int(len(pts)),
        'n_holes': int(pts[COLS['hole']].nunique()),
        'extent': {'x': [_num(coords[:, 0].min()), _num(coords[:, 0].max())],
                   'y': [_num(coords[:, 1].min()), _num(coords[:, 1].max())],
                   'z': [_num(coords[:, 2].min()), _num(coords[:, 2].max())]},
        'interval_m': _stats(interval),
        'assay': {'column': cfg['score_columns'][0],
                  **_stats(pts[cfg['score_columns'][0]])},
        # min_neighbors is an absolute sample count, so what it actually asks
        # for moves with the interval length. Reported, not adjusted: the
        # geologist scales the threshold, the app just shows the number.
        'support': {'min_neighbors': int(cfg['min_neighbors']),
                    'median_interval_m': _num(median_interval),
                    'effective_core_m': _num(cfg['min_neighbors'] * median_interval)},
        'search_radius_m': _num(cfg['radius_m']),
        'rim': _rim(coords, cfg['radius_m']),
        'active_curve': _active_curve(pts, cfg),
    }


def _active_curve(pts: pd.DataFrame, cfg: dict) -> dict:
    """How many samples survive each grade cutoff, at one percent steps.

    Sent once so the cutoff slider responds instantly instead of asking the
    engine on every drag. Computed rather than assumed: with assays piled up at
    detection limit the percentile ranks form plateaus, so the count steps
    rather than sliding linearly, and telling the user otherwise would be
    quietly wrong.
    """
    scored = build_mineralization_score(
        pts, cfg['score_columns'], log_transform=cfg['log_transform_score'],
        active_quantile=cfg['active_score_quantile'])
    rank = scored['score_rank'].to_numpy(float)

    steps = np.round(np.arange(0, 101) / 100.0, 2)
    counts = [int((rank >= q).sum()) for q in steps]
    return {'quantiles': [float(q) for q in steps], 'counts': counts}


def _rim(coords: np.ndarray, radius_m: float) -> dict:
    """How much of this file sits within one search radius of its own edge.

    Those samples have neighbours on the inward side only, so their
    orientations lean toward the cut face. When the spatial filter happens
    upstream the missing neighbours are simply not in the file and nothing
    downstream can recover them -- so the honest move is to measure the
    exposure and say so, rather than to imply the whole volume is equally good.
    """
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    near_edge = ((coords - lo) < radius_m) | ((hi - coords) < radius_m)

    # Only the horizontal extent is judged: a drillhole dataset is almost
    # always thinner than one radius in Z, which would flag every sample.
    exposed = near_edge[:, :2].any(axis=1)
    return {'radius_m': _num(radius_m),
            'n_exposed': int(exposed.sum()),
            'pct_exposed': _num(100.0 * exposed.mean()),
            'span_x_m': _num(hi[0] - lo[0]),
            'span_y_m': _num(hi[1] - lo[1])}


def _panels(ds, cfg: dict, estimator: str) -> dict:
    """The chosen estimator and the drill-pattern reference, on shared edges."""
    graded = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code, cfg)
    ungraded = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code,
                              cfg, use_score_weight=False)

    carry = [c for c in ['src_index', COLS['hole'], COLS['from'], COLS['to'],
                         'mid_m', 'mid_x', 'mid_y', 'mid_z', 'min_score',
                         'interval_m'] if c in ds.active.columns]
    out = {}

    for name in (estimator, REFERENCE):
        edges = ungraded if name == REFERENCE else graded
        fn = shape_pca if name == REFERENCE else ESTIMATORS[name]

        t0 = time.perf_counter()
        field = field_to_orientations(fn(edges, ds.coords, ds.scores,
                                         ds.hole_code, cfg))
        elapsed = time.perf_counter() - t0

        df = pd.concat([ds.active[carry].reset_index(drop=True), field], axis=1)
        df['node'] = np.arange(len(df))
        df['local_n'] = edges.neighbor_counts()
        df['local_holes'] = edges.hole_counts(ds.hole_code)
        df = df.loc[df['valid']].drop(columns='valid').reset_index(drop=True)
        df.attrs['elapsed_s'] = elapsed
        out[name] = (df, edges)

    return out


def _why_empty(ds, cfg: dict) -> dict:
    """Name the gate that closed, rather than returning a blank field."""
    edges = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code, cfg)
    counts = edges.neighbor_counts()
    holes = edges.hole_counts(ds.hole_code)
    return {
        'n_active': int(len(ds.coords)),
        'min_neighbors': int(cfg['min_neighbors']),
        'neighbors_median': _num(np.median(counts)) if len(counts) else None,
        'neighbors_max': int(counts.max()) if len(counts) else 0,
        'failed_on_neighbors': int((counts < cfg['min_neighbors']).sum()),
        'min_holes': int(cfg.get('min_holes', 0)),
        'holes_median': _num(np.median(holes)) if len(holes) else None,
        'failed_on_holes': int((holes < cfg.get('min_holes', 0)).sum()),
    }


def run(payload: dict) -> dict:
    """Estimate the field, and everything needed to judge it."""
    if _STATE['points'] is None:
        raise RuntimeError("No data loaded. Call 'load' first.")

    cfg = _cfg_from(payload)
    estimator = payload.get('estimator') or cfg['estimator']
    _STATE['cfg'] = cfg

    t0 = time.perf_counter()
    ds = prepare_points(_STATE['points'], cfg)
    prep_s = time.perf_counter() - t0

    if not len(ds.coords):
        raise ValueError(
            "No samples are active. active_score_quantile "
            f"({cfg['active_score_quantile']}) selected nothing.")

    panels = _panels(ds, cfg, estimator)
    method, graded_edges = panels[estimator]
    reference, ungraded_edges = panels[REFERENCE]

    if not len(method):
        return {'ok': False, 'empty': True, 'diagnosis': _why_empty(ds, cfg)}

    poles = ['pole_vec_x', 'pole_vec_y', 'pole_vec_z']

    # Angle to the drill pattern at every node both panels answered for. NaN
    # elsewhere, so the array stays aligned with the method's own rows.
    ref_by_node = reference.set_index('node')
    shared = method['node'].isin(ref_by_node.index).to_numpy()
    vs_ref = np.full(len(method), np.nan)
    if shared.any():
        vs_ref[shared] = angular_difference(
            method.loc[shared, poles].to_numpy(),
            ref_by_node.loc[method.loc[shared, 'node'], poles].to_numpy())

    _STATE.update(ds=ds, result=method, vs_ref=vs_ref)

    diagnostics = {
        'vs_reference_deg': _num(np.nanmedian(vs_ref)),
        'coherence_deg': _num(np.nanmedian(neighbor_agreement(
            method[poles].to_numpy(), graded_edges, method['node'].to_numpy()))),
        'coherence_null_deg': _num(coherence_null(
            method[poles].to_numpy(), graded_edges, method['node'].to_numpy())),
    }

    if payload.get('with_stability', True):
        stab, n_stab = split_half_stability(
            ESTIMATORS[estimator], ds.coords, ds.scores, ds.lengths,
            ds.hole_code, cfg)
        diagnostics.update(stability_deg=_num(stab), n_stability=int(n_stab))

    # The bundled sample carries analytic ground truth; a real file will not.
    true_err = None
    if 'true_pole_x' in ds.active.columns:
        truth = ds.active.loc[method['node'].to_numpy(),
                              ['true_pole_x', 'true_pole_y', 'true_pole_z']]
        true_err = _num(np.nanmedian(angular_difference(
            method[poles].to_numpy(), truth.to_numpy())))

    counts = graded_edges.neighbor_counts()
    holes = graded_edges.hole_counts(ds.hole_code)

    return {
        'ok': True,
        'estimator': estimator,
        'n_active': int(len(ds.coords)),
        # What the validity gates actually saw. Reported on success too, not
        # only on failure: a run that keeps 60% of its nodes is worth the same
        # explanation as one that keeps none.
        'neighbors': {'median': _num(np.median(counts)) if len(counts) else None,
                      'p10': _num(np.percentile(counts, 10)) if len(counts) else None,
                      'max': int(counts.max()) if len(counts) else 0,
                      'holes_median': _num(np.median(holes)) if len(holes) else None,
                      'failed_on_neighbors': int((counts < cfg['min_neighbors']).sum()),
                      'failed_on_holes': int((holes < cfg.get('min_holes', 0)).sum())},
        'n_valid': int(len(method)),
        'pct_active': _num(100.0 * len(method) / max(len(ds.coords), 1)),
        'timing_s': {'prepare': _num(prep_s),
                     'method': _num(method.attrs.get('elapsed_s')),
                     'reference': _num(reference.attrs.get('elapsed_s'))},
        'diagnostics': diagnostics,
        'true_error_deg': true_err,
        'geometry': _geometry(ds, method, reference, vs_ref),
    }


def _geometry(ds, method: pd.DataFrame, reference: pd.DataFrame,
              vs_ref: np.ndarray) -> dict:
    """Bulk arrays for the 3D view and the stereonet, as transferable bytes."""
    coords = method[['mid_x', 'mid_y', 'mid_z']].to_numpy(float)
    origin = ds.coords.min(axis=0)

    traces, offsets, names = [], [0], []
    for hole, path in ds.paths.items():
        traces.append(path[['X', 'Y', 'Z']].to_numpy(float) - origin)
        offsets.append(offsets[-1] + len(path))
        names.append(str(hole))

    trace_xyz = (np.concatenate(traces) if traces
                 else np.empty((0, 3), dtype=float))

    return {
        'n': int(len(method)),
        'origin': [float(v) for v in origin],
        'coords': _f32(coords - origin),
        'poles': _f32(method[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()),
        # The reference has its own validity gate, so its rows do not line up
        # with the method's. Kept in its own block with its own length rather
        # than silently mismatching the arrays above.
        'reference': {
            'n': int(len(reference)),
            'trend': _f32(reference['pole_trend_deg'].to_numpy()),
            'plunge': _f32(reference['pole_plunge_deg'].to_numpy()),
        },
        'dip': _f32(method['plane_dip_deg'].to_numpy()),
        'dip_azimuth': _f32((method['plane_strike_deg'].to_numpy() + 90.0) % 360.0),
        'trend': _f32(method['pole_trend_deg'].to_numpy()),
        'plunge': _f32(method['pole_plunge_deg'].to_numpy()),
        'planarity': _f32(method['planarity_norm'].to_numpy()),
        'score': _f32(method['min_score'].to_numpy()),
        'vs_reference': _f32(vs_ref),
        'hole_index': _i32(pd.factorize(method[COLS['hole']].astype(str))[0]),
        'trace_xyz': _f32(trace_xyz),
        'trace_offsets': _i32(offsets),
        'trace_names': names,
    }


def export(payload: dict) -> dict:
    """The Leapfrog CSV, as text. Writing it is the browser's job, not ours."""
    if _STATE['result'] is None:
        raise RuntimeError("Nothing to export. Call 'run' first.")

    table = to_leapfrog(_STATE['result'], vs_reference_deg=_STATE['vs_ref'])
    buf = io.StringIO()
    table.to_csv(buf, index=False)
    return {'csv': buf.getvalue(), 'n_rows': int(len(table)),
            'columns': list(table.columns)}


_ACTIONS = {'versions': versions, 'preview': preview, 'load': load,
            'run': run, 'export': export}


def handle(action: str, payload_json: str = '{}'):
    """Single entry point. Never raises into JavaScript; errors come back as data."""
    try:
        fn = _ACTIONS[action]
    except KeyError:
        return {'ok': False, 'error': f"Unknown action {action!r}",
                'error_type': 'UnknownAction'}

    try:
        result = fn(json.loads(payload_json or '{}'))
        result.setdefault('ok', True)
        return result
    except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
        return {'ok': False, 'error': str(exc), 'error_type': type(exc).__name__,
                'traceback': ''.join(traceback.format_exception(exc))[-2000:]}
