"""Estimator bake-off: shape-PCA vs graph structure tensor vs edge tensor.

All estimators run over one prepared dataset and one shared neighbourhood
graph, so the comparison is of estimators, not of neighbourhoods.

A fourth panel, GEOMETRY CONTROL, runs shape-PCA on weights that carry no
grade information at all -- pure distance and interval length. It is what the
drill pattern alone reports. An estimator whose stereonet resembles the control
is measuring drilling, not geology. That comparison is the point of the whole
exercise (README section 2), so it is reported as a first-class number:
`vs_control_deg`, the median angle between an estimator's poles and the
control's poles at the same nodes.

Usage:
    python scripts/bakeoff.py                    # whole property
    python scripts/bakeoff.py --prospect "Tom West"
    python scripts/bakeoff.py --all-prospects
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import make_cfg
from core.estimators import ESTIMATORS, field_to_orientations, shape_pca
from core.geometry import angular_difference, woodcock
from core.ingest import load_tables
from core.neighbors import build_edge_set
from core.pipeline import prepare
from core.schema import COLS
from core.validation import (coherence_null, neighbor_agreement,
                             split_half_stability)

DATA = {
    'samples': 'MPA_Samples_BD_20240227.csv',
    'collars': 'MPA_Collar_20240227.csv',
    'surveys': 'MPA_Survey_20240227.csv',
}
ORDER = ['geometry_control', 'shape_pca', 'structure_tensor', 'edge_tensor',
         'lsq_gradient']
OUT = Path('out')


def _poles(df: pd.DataFrame) -> np.ndarray:
    return df[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()


def _extra_diagnostics(ds, cfg, results):
    """Spatial coherence and split-half stability -- the tests that separate
    'different from the drilling' from 'noise'."""
    graded = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code, cfg)
    ungraded = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code,
                              cfg, use_score_weight=False)
    diag = {}
    for name, df in results.items():
        edges = ungraded if name == 'geometry_control' else graded
        nodes = df['node'].to_numpy()
        poles = _poles(df)

        coh = float(np.nanmedian(neighbor_agreement(poles, edges, nodes)))
        null = coherence_null(poles, edges, nodes)

        fn = shape_pca if name == 'geometry_control' else ESTIMATORS[name]
        stab, n_stab = split_half_stability(
            fn, ds.coords, ds.scores, ds.lengths, ds.hole_code, cfg,
            use_score_weight=(name != 'geometry_control'))

        diag[name] = {'coherence_deg': coh, 'coherence_null_deg': null,
                      'stability_deg': stab, 'n_stability': n_stab}
    return diag


def run_all(ds, cfg) -> dict[str, pd.DataFrame]:
    """Every estimator plus the geometry control, on shared edges."""
    graded = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code, cfg)
    ungraded = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code,
                              cfg, use_score_weight=False)

    carry = [c for c in ['src_index', COLS['hole'], 'mid_m', 'mid_x', 'mid_y',
                         'mid_z', 'min_score'] if c in ds.active.columns]
    results = {}

    for name in ORDER:
        edges = ungraded if name == 'geometry_control' else graded
        fn = shape_pca if name == 'geometry_control' else ESTIMATORS[name]

        t0 = time.perf_counter()
        tf = fn(edges, ds.coords, ds.scores, ds.hole_code, cfg)
        orient = field_to_orientations(tf)
        elapsed = time.perf_counter() - t0

        out = pd.concat([ds.active[carry].reset_index(drop=True), orient], axis=1)
        out['node'] = np.arange(len(out))
        out['local_n'] = edges.neighbor_counts()
        out['local_holes'] = edges.hole_counts(ds.hole_code)
        out['estimator'] = name
        out.attrs['elapsed_s'] = elapsed
        results[name] = out.loc[out['valid']].drop(columns='valid').reset_index(drop=True)
        results[name].attrs['elapsed_s'] = elapsed

    return results


def summarize(results: dict[str, pd.DataFrame], n_active: int,
              diag: dict | None = None) -> pd.DataFrame:
    """One row per estimator. Higher S1/C = tighter pole concentration."""
    control = results['geometry_control'].set_index('node')
    rows = []

    for name in ORDER:
        df = results[name]
        conf_cut = df['planarity_norm'].quantile(0.75)
        conf = df.loc[df['planarity_norm'] >= conf_cut]

        w_all = woodcock(_poles(df))
        w_conf = woodcock(_poles(conf))

        shared = df.set_index('node').index.intersection(control.index)
        if name == 'geometry_control' or len(shared) == 0:
            vs_control = np.nan
        else:
            vs_control = float(np.median(angular_difference(
                _poles(df.set_index('node').loc[shared]),
                _poles(control.loc[shared]))))

        rows.append({
            'estimator': name,
            'n_valid': len(df),
            'pct_active': 100.0 * len(df) / max(n_active, 1),
            'secs': df.attrs.get('elapsed_s', np.nan),
            'planarity_norm_med': df['planarity_norm'].median(),
            'linearity_norm_med': df['linearity_norm'].median(),
            'S1_all': w_all['S1'], 'C_all': w_all['C'], 'K_all': w_all['K'],
            'S1_conf': w_conf['S1'], 'C_conf': w_conf['C'],
            'vs_control_deg': vs_control,
            **(diag or {}).get(name, {}),
        })
    return pd.DataFrame(rows)


def plot(results: dict[str, pd.DataFrame], title: str, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from viz.stereonet import density_contour, draw_frame

    fig, axes = plt.subplots(2, len(ORDER), figsize=(3.5 * len(ORDER), 8.8))
    for col, name in enumerate(ORDER):
        df = results[name]
        conf = df.loc[df['planarity_norm'] >= df['planarity_norm'].quantile(0.75)]
        w = woodcock(_poles(conf))

        ax = axes[0, col]
        density_contour(ax, conf['pole_trend_deg'], conf['pole_plunge_deg'])
        draw_frame(ax, f"{name}\npoles (top-quartile confidence)",
                   f"n={len(conf):,}   S1={w['S1']:.3f}   C={w['C']:.2f}")

        ax = axes[1, col]
        density_contour(ax, df['line_trend_deg'], df['line_plunge_deg'],
                        cmap='viridis')
        draw_frame(ax, 'lineations (all valid)', f"n={len(df):,}")

    fig.suptitle(title, fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.4)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def bakeoff(label: str, cfg: dict, tables) -> pd.DataFrame:
    samples, collars, surveys = tables
    ds = prepare(samples, collars, surveys, cfg)
    if len(ds.active) < cfg['min_neighbors']:
        print(f"  [{label}] only {len(ds.active)} active samples -- skipped")
        return pd.DataFrame()

    results = run_all(ds, cfg)
    diag = _extra_diagnostics(ds, cfg, results)
    summary = summarize(results, len(ds.active), diag)
    summary.insert(0, 'subset', label)

    slug = label.lower().replace(' ', '_')
    OUT.mkdir(exist_ok=True)
    plot(results, f"MacPass -- {label}   "
                  f"(r={cfg['radius_m']:.0f} m, min_n={cfg['min_neighbors']}, "
                  f"active={len(ds.active):,})",
         OUT / f'bakeoff_{slug}.png')
    for name, df in results.items():
        df.to_csv(OUT / f'orientations_{slug}_{name}.csv', index=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prospect', default=None)
    ap.add_argument('--all-prospects', action='store_true')
    ap.add_argument('--radius', type=float, default=120.0)
    ap.add_argument('--min-neighbors', type=int, default=100)
    args = ap.parse_args()

    cfg = make_cfg(radius_m=args.radius, min_neighbors=args.min_neighbors)
    samples, collars, surveys = load_tables(
        DATA['samples'], DATA['collars'], DATA['surveys'], cfg)
    print(f"loaded: {len(samples):,} samples / {len(collars):,} collars / "
          f"{len(surveys):,} survey rows")

    subsets = [('all', None)]
    if args.prospect:
        subsets = [(args.prospect, args.prospect)]
    elif args.all_prospects:
        counts = (samples.merge(collars[[COLS['hole'], COLS['prospect']]],
                                on=COLS['hole'], how='left')
                  .groupby(COLS['prospect']).size().sort_values(ascending=False))
        subsets += [(p, p) for p in counts[counts >= 1500].index]

    frames = []
    for label, prospect in subsets:
        print(f"\n=== {label} ===")
        sub_cfg = make_cfg(radius_m=args.radius, min_neighbors=args.min_neighbors,
                           prospect_filter=prospect)
        from core.ingest import apply_filters
        tables = apply_filters(samples, collars, surveys, collars, sub_cfg)
        summary = bakeoff(label, sub_cfg, tables)
        if not summary.empty:
            fmt = lambda v: f"{v:8.3f}"
            print("\n  -- field & stereonet --")
            print(summary[['estimator', 'n_valid', 'secs', 'planarity_norm_med',
                           'S1_all', 'C_all', 'S1_conf', 'C_conf']]
                  .to_string(index=False, float_format=fmt))
            print("\n  -- is it geology? (lower coherence/stability = better; "
                  "~60 deg = no signal) --")
            print(summary[['estimator', 'vs_control_deg', 'coherence_deg',
                           'coherence_null_deg', 'stability_deg', 'n_stability']]
                  .to_string(index=False, float_format=fmt))
            frames.append(summary)

    if frames:
        allsum = pd.concat(frames, ignore_index=True)
        allsum.to_csv(OUT / 'bakeoff_summary.csv', index=False)
        print(f"\nwrote {OUT/'bakeoff_summary.csv'} and per-subset figures")


if __name__ == '__main__':
    main()
