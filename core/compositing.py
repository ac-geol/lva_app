"""Downhole compositing: change the support of the assay data before differencing.

Every estimator here works from finite differences of the score between holes,

    delta_ij = (s_j - s_i) / d_ij,

so each node's noise enters the fit as sigma_eps / d. The baseline d is already
as long as the geology allows (radius_m = 120), which leaves sigma_eps as the
only remaining lever -- and MacPass assays sit at ~1.3 m support, where an
individual interval is mostly nugget. Averaging to a longer support attacks the
noise at source. See docs/BAKEOFF.md: the synthetic benchmark puts the estimator
at 9 degrees clean and 32 degrees at noise 0.5, and the real data at 27 degrees.

This is NOT the neighbourhood smoothing that `smoothing.smooth_scores` performs
and that the tuning sweep showed to be actively harmful. That averages across
the 3-D graph whose edges are then differenced, so each node's value bleeds into
its neighbours and `s_j - s_i` becomes partly a difference of shared terms --
manufacturing agreement between exactly the pair being measured. Compositing
averages along the hole only, before the graph exists; with same-hole edges
excluded, i and j always sit in different holes, so no term is ever shared
across a difference. The support changes; the neighbours stay independent.

The genuine cost is that structure thinner than the composite length is
averaged away, so the useful range is bounded above by the thickness of the
thing being resolved.

Compositing runs on RAW GRADES, never on the mineralization score: the score is
mean(z(log1p(grade))) and log of a mean is not the mean of a log, so compositing
afterwards would silently change what is being averaged. Metal content is
conserved -- composite values are length-weighted means, with source intervals
split proportionally across composite boundaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COLS


def _run_ids(hole_code: np.ndarray, frm: np.ndarray, to: np.ndarray,
             gap_tolerance_m: float) -> np.ndarray:
    """Contiguous downhole runs. A hole change or a wide gap starts a new one.

    Gaps narrower than the tolerance stay *inside* a composite and are simply
    unassayed material: they lower the composite's assayed length rather than
    splitting it. Wide gaps are separate sampled zones -- MacPass has 959 real
    gaps, the largest 335 m -- and compositing across one would invent a value
    for ground nobody assayed.
    """
    new_hole = np.r_[True, hole_code[1:] != hole_code[:-1]]
    gap = np.r_[0.0, frm[1:] - to[:-1]]
    return np.cumsum(new_hole | (gap > gap_tolerance_m)) - 1


def composite_downhole(samples: pd.DataFrame, length_m: float | None, *,
                       value_columns, gap_tolerance_m: float | None = None,
                       min_tail_fraction: float = 0.5,
                       min_coverage: float = 0.0) -> pd.DataFrame:
    """Composite `samples` to a fixed downhole length.

    `length_m` of None or <= 0 returns the input unchanged, so compositing is
    strictly opt-in and the uncomposited pipeline is bit-for-bit unaffected.

    Composites are laid out on a fixed grid from the start of each run. A source
    interval straddling a boundary contributes its overlapping length to both
    sides. Values are length-weighted means over the intervals that actually
    carry a value for that column, so a missing assay reduces the weight behind
    a composite rather than poisoning it. That value then stands for the
    composite's whole assayed length -- unassayed ground is treated as
    represented by the material around it, never as zero grade. Metal is
    conserved exactly across the intervals that carry an assay; a column with
    many nulls (MacPass Ag: 1113 of 32546) redistributes a little of it into
    the gaps those nulls leave.

    Parameters
    ----------
    value_columns : the raw grade columns to average. Anything else is dropped.
    gap_tolerance_m : gaps wider than this split a run. None => `length_m`,
        i.e. a gap shorter than one composite does not break contiguity.
    min_tail_fraction : a run's final cell shorter than this fraction of
        `length_m` is merged into the preceding composite instead of standing as
        an undersized one. A run shorter than `length_m` is always kept whole.
    min_coverage : drop composites whose assayed length is below this fraction
        of `length_m`. 0.0 disables the filter -- the honest default, since a
        sparsely covered composite is visible in the `coverage` column and the
        right cutoff is a per-dataset calibration.

    Returns
    -------
    A samples-shaped frame carrying HoleID / From_m / To_m / interval_m / mid_m
    plus the value columns, ready for `desurvey_samples`.

    `From_m` and `To_m` bound the assayed material, while `interval_m` is the
    assayed length actually behind the value. These differ whenever a composite
    spans an internal gap, and `interval_m` is the one that matters: it is the
    support, and `neighbors.edge_weights` weights a neighbour by it.
    """
    if length_m is None or length_m <= 0:
        return samples

    hole, c_from, c_to = COLS['hole'], COLS['from'], COLS['to']
    value_columns = [c for c in value_columns if c in samples.columns]
    if not value_columns:
        raise ValueError(
            f"None of the requested value columns are present. "
            f"Available (first 30): {list(samples.columns)[:30]}")

    if gap_tolerance_m is None:
        gap_tolerance_m = float(length_m)

    df = samples.sort_values([hole, c_from], kind='mergesort').reset_index(drop=True)
    frm = df[c_from].to_numpy(float)
    to = df[c_to].to_numpy(float)
    hole_code, hole_labels = pd.factorize(df[hole].astype(str))

    run = _run_ids(hole_code, frm, to, gap_tolerance_m)
    n_runs = int(run.max()) + 1 if len(run) else 0
    if n_runs == 0:
        return samples.iloc[0:0].copy()

    # Each run's grid is anchored at its own first interval.
    run_start = np.full(n_runs, np.inf)
    np.minimum.at(run_start, run, frm)
    run_end = np.full(n_runs, -np.inf)
    np.maximum.at(run_end, run, to)

    # Split every source interval across the composite boundaries it spans.
    eps = 1e-9 * length_m
    k_lo = np.floor((frm - run_start[run]) / length_m).astype(np.int64)
    k_hi = np.floor((to - run_start[run] - eps) / length_m).astype(np.int64)
    k_hi = np.maximum(k_hi, k_lo)
    n_pieces = k_hi - k_lo + 1

    src = np.repeat(np.arange(len(df), dtype=np.int64), n_pieces)
    offset = np.arange(len(src)) - np.repeat(
        np.r_[0, np.cumsum(n_pieces)[:-1]], n_pieces)
    k = k_lo[src] + offset
    p_run = run[src]

    cell_lo = run_start[p_run] + k * length_m
    p_from = np.maximum(frm[src], cell_lo)
    p_to = np.minimum(to[src], cell_lo + length_m)
    p_len = p_to - p_from
    keep = p_len > 0
    src, k, p_run, p_from, p_to, p_len = (
        a[keep] for a in (src, k, p_run, p_from, p_to, p_len))

    # A stub final cell joins the previous composite rather than standing as an
    # undersized one. Runs of a single cell have nothing to merge into.
    k_max = np.zeros(n_runs, dtype=np.int64)
    np.maximum.at(k_max, p_run, k)
    tail_len = run_end - (run_start + k_max * length_m)
    merge = (tail_len < min_tail_fraction * length_m) & (k_max > 0)
    is_tail = merge[p_run] & (k == k_max[p_run])
    k = np.where(is_tail, k - 1, k)

    gid, _ = pd.factorize(p_run.astype(np.int64) * (int(k.max()) + 2) + k)
    n_out = int(gid.max()) + 1

    assayed = np.bincount(gid, weights=p_len, minlength=n_out)
    out_from = np.full(n_out, np.inf)
    np.minimum.at(out_from, gid, p_from)
    out_to = np.full(n_out, -np.inf)
    np.maximum.at(out_to, gid, p_to)
    p_mid = 0.5 * (p_from + p_to)
    centroid = np.bincount(gid, weights=p_len * p_mid, minlength=n_out) / assayed

    out_hole = np.empty(n_out, dtype=np.int64)
    out_hole[gid] = hole_code[src]

    out = pd.DataFrame({
        hole: hole_labels[out_hole],
        c_from: out_from,
        c_to: out_to,
        'interval_m': assayed,
        'mid_m': centroid,
        'n_samples': np.bincount(gid, minlength=n_out),
        'coverage': assayed / float(length_m),
    })

    # Length-weighted mean per column, over the intervals that carry a value.
    for col in value_columns:
        v = pd.to_numeric(df[col], errors='coerce').to_numpy(float)[src]
        has = np.isfinite(v)
        w = np.where(has, p_len, 0.0)
        num = np.bincount(gid, weights=w * np.where(has, v, 0.0), minlength=n_out)
        den = np.bincount(gid, weights=w, minlength=n_out)
        out[col] = np.divide(num, den, out=np.full(n_out, np.nan), where=den > 0)

    if min_coverage > 0:
        out = out.loc[out['coverage'] >= min_coverage]

    return (out.sort_values([hole, c_from], kind='mergesort')
               .reset_index(drop=True))
