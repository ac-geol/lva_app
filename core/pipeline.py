"""End-to-end orchestration: tables in, orientation field out."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .compositing import composite_downhole
from .desurvey import desurvey_samples
from .estimators import ESTIMATORS, field_to_orientations
from .neighbors import build_edge_set
from .schema import COLS
from .score import build_mineralization_score


@dataclass
class Dataset:
    """Everything downstream of ingest that is estimator-independent."""
    des: pd.DataFrame            # all desurveyed, scored samples
    active: pd.DataFrame         # the subset used as estimation nodes
    coords: np.ndarray
    scores: np.ndarray
    lengths: np.ndarray
    hole_code: np.ndarray
    hole_ids: np.ndarray
    paths: dict


def prepare(samples, collars, surveys, cfg) -> Dataset:
    # Compositing precedes desurvey and scoring: it is a downhole operation on
    # raw grades, and the score's log1p means it cannot be applied afterwards.
    samples = composite_downhole(
        samples, cfg.get('composite_length_m'),
        value_columns=cfg['score_columns'],
        gap_tolerance_m=cfg.get('composite_gap_tolerance_m'),
        min_tail_fraction=cfg.get('composite_min_tail_fraction', 0.5),
        min_coverage=cfg.get('composite_min_coverage', 0.0))

    des, paths = desurvey_samples(samples, collars, surveys)
    des = build_mineralization_score(
        des, cfg['score_columns'],
        log_transform=cfg['log_transform_score'],
        active_quantile=cfg['active_score_quantile'])

    active = des.loc[des['is_active']].copy().reset_index(drop=True)
    hole_ids = active[COLS['hole']].astype(str).to_numpy()
    codes, _ = pd.factorize(hole_ids)

    return Dataset(
        des=des, active=active,
        coords=active[['mid_x', 'mid_y', 'mid_z']].to_numpy(float),
        scores=active['min_score'].to_numpy(float),
        lengths=active['interval_m'].to_numpy(float),
        hole_code=codes.astype(np.int64), hole_ids=hole_ids, paths=paths)


def run_estimator(ds: Dataset, cfg: dict, name: str | None = None,
                  edges=None) -> pd.DataFrame:
    """Run one estimator over a prepared dataset. Returns valid rows only."""
    name = name or cfg['estimator']
    if name not in ESTIMATORS:
        raise KeyError(f"Unknown estimator {name!r}. "
                       f"Choose from {sorted(ESTIMATORS)}")

    if edges is None:
        edges = build_edge_set(ds.coords, ds.scores, ds.lengths,
                               ds.hole_code, cfg)

    tf = ESTIMATORS[name](edges, ds.coords, ds.scores, ds.hole_code, cfg)
    orient = field_to_orientations(tf)

    carry = ['src_index', COLS['hole'], COLS['from'], COLS['to'], 'mid_m',
             'mid_x', 'mid_y', 'mid_z', 'min_score', 'interval_m']
    carry = [c for c in carry if c in ds.active.columns]
    out = pd.concat([ds.active[carry].reset_index(drop=True), orient], axis=1)

    out['local_n'] = edges.neighbor_counts()
    out['local_holes'] = edges.hole_counts(ds.hole_code)
    return out.loc[out['valid']].drop(columns='valid').reset_index(drop=True)
