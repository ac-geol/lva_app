"""Judge each LVA method against surfaces derived from logged geology.

The external check described in CLAUDE.md section 4 -- the one metric here
that is not computed from the same graph it judges.

Reference surfaces come from the logged-interval table: one contact point per
hole per named stratigraphic unit, plane-fitted. Not oriented core, but real, independent of
the assay data, and free of downhole collinearity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import make_cfg
from core.contacts import compare_to_planes, extract_contacts, fit_contact_planes
from core.estimators import ESTIMATORS, field_to_orientations, shape_pca
from core.ingest import load_tables
from core.neighbors import build_edge_set
from core.pipeline import prepare
from core.schema import COLS

# Bring your own tables. `data/` is gitignored for exactly this; edit these
# paths to match your filenames.
SAMPLES = 'data/samples.csv'
COLLARS = 'data/collars.csv'
SURVEYS = 'data/surveys.csv'
INTERP = 'data/logged_intervals.csv'

# Same panel set as scripts/compare_methods.py: two methods, then the
# drill-pattern reference (shape-PCA on grade-blind weights).
METHODS = ['shape_pca', 'lsq_gradient']
REFERENCE = 'geometry_control'
PANELS = METHODS + [REFERENCE]


def main():
    cfg = make_cfg()
    samples, collars, surveys = load_tables(SAMPLES, COLLARS, SURVEYS, cfg)
    interp = pd.read_csv(INTERP)

    contacts = extract_contacts(interp, collars, surveys)
    planes = fit_contact_planes(contacts, min_holes=8)
    Path('out').mkdir(exist_ok=True)
    contacts.to_csv('out/contacts.csv', index=False)
    planes.to_csv('out/contact_planes.csv', index=False)

    pd.set_option('display.width', 220)
    print(f"contact points: {len(contacts):,} over {contacts['code'].nunique()} codes")
    print(f"fitted surfaces: {len(planes)}  ({int(planes['usable'].sum())} usable)\n")
    print(planes.loc[planes['usable'],
                     ['code', 'boundary', 'n_holes', 'strike_deg', 'dip_deg',
                      'rms_resid_m', 'planarity', 'collar_aspect', 'extent_m']]
          .head(15).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    rejected = planes.loc[~planes['usable']]
    if len(rejected):
        print(f"\nrejected as single-fence (collar_aspect < 0.15): "
              f"{', '.join(rejected['code'] + '/' + rejected['boundary'])}")

    ds = prepare(samples, collars, surveys, cfg)
    graded = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code, cfg)
    ungraded = build_edge_set(ds.coords, ds.scores, ds.lengths, ds.hole_code,
                              cfg, use_score_weight=False)

    print("\n=== method vs logged contact surfaces ===")
    summary = []
    for name in PANELS:
        edges = ungraded if name == REFERENCE else graded
        fn = shape_pca if name == REFERENCE else ESTIMATORS[name]
        tf = fn(edges, ds.coords, ds.scores, ds.hole_code, cfg)
        out = field_to_orientations(tf)
        out = pd.concat([ds.active[['mid_x', 'mid_y', 'mid_z']].reset_index(drop=True),
                         out], axis=1)
        out = out.loc[out['valid']].reset_index(drop=True)

        cmp = compare_to_planes(out, planes, contacts)
        if cmp.empty:
            continue
        cmp.insert(0, 'method', name)
        summary.append(cmp)

    allcmp = pd.concat(summary, ignore_index=True)
    allcmp.to_csv('out/contact_validation.csv', index=False)

    pivot = allcmp.pivot_table(index=['code', 'boundary', 'n_holes', 'n_nodes',
                                      'ref_strike', 'ref_dip'],
                               columns='method', values='median_err_deg')
    pivot = pivot[[c for c in PANELS if c in pivot.columns]]
    print(pivot.round(1).to_string())

    print("\n=== overall (node-weighted median error vs reference, degrees) ===")
    agg = (allcmp.assign(wt=lambda d: d['n_nodes'])
           .groupby('method')
           .apply(lambda g: np.average(g['median_err_deg'], weights=g['wt']),
                  include_groups=False)
           .reindex(PANELS))
    print(agg.round(1).to_string())
    print("\nreference: 60 deg is the expectation for random axes.")


if __name__ == '__main__':
    main()
