"""End-to-end orchestration: tables in, orientation field out."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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


def hole_paths_from_points(points: pd.DataFrame) -> dict:
    """Reconstruct a drawable trace per hole from desurveyed sample midpoints.

    Coordinates only: a point table carries no survey attitude, so unlike
    `desurvey.build_hole_path` there are no downhole direction columns. This
    is enough to draw the hole alongside its orientation discs, which is all
    anything downstream asks of it.
    """
    out = {}
    for hole_id, g in points.groupby(COLS['hole'], sort=False):
        g = g.sort_values('mid_m')
        out[str(hole_id)] = pd.DataFrame({
            COLS['depth']: g['mid_m'].to_numpy(float),
            'X': g['mid_x'].to_numpy(float),
            'Y': g['mid_y'].to_numpy(float),
            'Z': g['mid_z'].to_numpy(float),
        }).reset_index(drop=True)
    return out


def prepare_points(points: pd.DataFrame, cfg: dict) -> Dataset:
    """Prepare an already-desurveyed point table. No collars, no surveys.

    Identical to `prepare` from scoring onward -- same score, same active
    selection, same Dataset -- so every estimator, validation statistic and
    diagnostic downstream cannot tell which route the coordinates arrived by.
    """
    des = points.copy()
    des[['mid_x', 'mid_y', 'mid_z']] = des[
        [COLS['easting'], COLS['northing'], COLS['elev']]].to_numpy(float)
    des = des.reset_index(drop=True).reset_index(names='src_index')

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
        hole_code=codes.astype(np.int64), hole_ids=hole_ids,
        paths=hole_paths_from_points(des))
