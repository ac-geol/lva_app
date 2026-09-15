"""A synthetic folded deposit with a known answer at every sample.

Why this exists
---------------
Every other check in this project is indirect. Split-half stability says a
field is reproducible, not that it is right. The logged contact surfaces in
`validate_contacts.py` are real but sparse, and they are an interpretation.
Neither can say "the true dip here is 42 degrees" -- so the honest reading of a
27-degree contact error has always been "27 degrees away from somebody's
interpretation", not "27 degrees wrong".

Here the geology is generated, so the answer is known analytically at every
sample. That turns method evaluation into a measurable error distribution
against a known quantity.

What it is NOT
--------------
**This validates the mathematics, not the geology.** A smooth analytic fold is
kinder than real rock: no faulting, no alteration overprint, stationary noise,
a mineralized shell of constant thickness that never pinches or splits. A
method can do well here and still do poorly on a real deposit.

Use it as a demo, a regression target, and a bias check. Do **not** select or
tune a method on it -- see `CLAUDE.md` section 4 for what
happened the last time a convenient internal metric drove selection.

The model
---------
A plunging antiform. The mineralized surface is

    f(x, y) = z0 + A * cos(2*pi*(x - x0) / wavelength) - (y - y0) * tan(plunge)

corrugated along x, with a fold axis running north and plunging gently. Grade
is a Gaussian shell about that surface, so the ore is *conformal to the fold*
and its orientation changes continuously across the deposit -- which is the
point. A planar body would be recovered by anything and would exercise nothing.

Because the shell is a constant vertical offset from f, every iso-surface
z = f(x, y) + c shares one normal at a given (x, y). The truth is therefore
exact for every sample in the shell, not just for those on the surface itself:

    n(x, y) = normalize( (A*k*sin(k*(x - x0)),  tan(plunge),  1) ),  k = 2*pi/wavelength

Coordinates are a deliberately fake local grid (origin 10000, 10000) so no
output of this module can be mistaken for a real survey.

Usage
-----
    from core.synthetic import make_example_deposit, example_cfg, orientation_error
    from core.ingest import load_tables
    from core.pipeline import prepare, run_estimator

    dep = make_example_deposit(seed=0)
    cfg = example_cfg()

    # load_tables accepts DataFrames as readily as paths, so the synthetic
    # tables enter through the same ingest path real CSVs do -- including the
    # derived interval_m / mid_m columns that prepare() expects.
    tables = load_tables(dep.samples, dep.collars, dep.surveys, cfg)
    ds = prepare(*tables, cfg)

    out = run_estimator(ds, cfg, 'lsq_gradient')
    err = orientation_error(out, ds.active)   # degrees per node, vs the truth
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import make_cfg
from .desurvey import build_hole_path
from .geometry import angular_difference, normal_to_strike_dip, unitize
from .schema import COLS

# Fake local grid. Not a real projection, and not near any real UTM zone.
ORIGIN_E = 10000.0
ORIGIN_N = 10000.0
COLLAR_Z = 1000.0

DEFAULTS = dict(
    # --- fold geometry (metres, degrees) ---
    z0=870.0,               # mean elevation of the mineralized surface
    amplitude=80.0,         # half the crest-to-trough relief
    wavelength=400.0,       # one full fold repeat along x
    plunge_deg=10.0,        # fold axis plunges this much towards +y (north)
    shell_thickness=25.0,   # Gaussian sigma of the ore shell, perpendicular

    # --- drill pattern ---
    n_sections=8,           # sections spaced along y (the fold axis)
    section_spacing=100.0,
    holes_per_section=14,
    hole_spacing=80.0,
    hole_length=300.0,
    collar_dip=-60.0,       # negative is downward, matching the tables
    sample_length=1.5,
    survey_interval=30.0,

    # --- grades ---
    cu_background=0.02,     # per cent
    cu_amplitude=1.80,
    au_background=0.01,     # grammes per tonne
    au_amplitude=2.20,
    cu_noise=0.25,          # lognormal sigma
    au_noise=0.60,          # gold is nuggetier, deliberately
    shared_noise=0.30,      # common factor: the two elements co-vary
)


@dataclass
class ExampleDeposit:
    """The three tables, plus the answer.

    The truth columns (`true_pole_x/y/z`, `true_strike_deg`, `true_dip_deg`)
    ride on `samples` as ordinary columns, so they survive desurvey and scoring
    and reappear on `Dataset.active`. `truth` is the same data on its own for
    inspection. Use `orientation_error(out, ds.active)` to score a run.
    """
    samples: pd.DataFrame
    collars: pd.DataFrame
    surveys: pd.DataFrame
    truth: pd.DataFrame
    params: dict = field(default_factory=dict)


def fold_surface(x, y, p: dict) -> np.ndarray:
    """Elevation of the mineralized surface at (x, y)."""
    k = 2.0 * np.pi / p['wavelength']
    return (p['z0']
            + p['amplitude'] * np.cos(k * (np.asarray(x, float) - ORIGIN_E))
            - (np.asarray(y, float) - ORIGIN_N) * np.tan(np.radians(p['plunge_deg'])))


def fold_normal(x, y, p: dict) -> np.ndarray:
    """Unit normal to the mineralized surface at (x, y). The ground truth.

    Exact for every point in the shell, not only on the surface: the shell is a
    constant vertical offset, so all its iso-surfaces share this normal.
    """
    k = 2.0 * np.pi / p['wavelength']
    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))
    # z = f(x, y)  ->  n proportional to (-df/dx, -df/dy, 1)
    dfdx = -p['amplitude'] * k * np.sin(k * (x - ORIGIN_E))
    dfdy = np.full_like(x, -np.tan(np.radians(p['plunge_deg'])))
    return unitize(np.column_stack([-dfdx, -dfdy, np.ones_like(x)]))


def _perpendicular_distance(x, y, z, p: dict) -> np.ndarray:
    """Signed perpendicular distance from a point to the mineralized surface."""
    k = 2.0 * np.pi / p['wavelength']
    dfdx = -p['amplitude'] * k * np.sin(k * (np.asarray(x, float) - ORIGIN_E))
    dfdy = -np.tan(np.radians(p['plunge_deg']))
    scale = np.sqrt(1.0 + dfdx ** 2 + dfdy ** 2)
    return (np.asarray(z, float) - fold_surface(x, y, p)) / scale


def _grades(x, y, z, p: dict, rng: np.random.Generator):
    """Cu and Au from distance to the shell, correlated, lognormally noisy."""
    d = _perpendicular_distance(x, y, z, p)
    shell = np.exp(-0.5 * (d / p['shell_thickness']) ** 2)

    n = len(shell)
    common = rng.lognormal(0.0, p['shared_noise'], n)
    cu = p['cu_background'] + p['cu_amplitude'] * shell * common \
        * rng.lognormal(0.0, p['cu_noise'], n)
    # Gold sits tighter to the core of the shell and is noisier on top of it.
    au = p['au_background'] + p['au_amplitude'] * shell ** 1.4 * common \
        * rng.lognormal(0.0, p['au_noise'], n)
    return np.round(cu, 4), np.round(au, 4)


def _drill_layout(p: dict, rng: np.random.Generator):
    """Collars and survey stations for a sectioned, deviated drill programme."""
    collar_rows, survey_rows = [], []

    for s in range(p['n_sections']):
        y = ORIGIN_N + s * p['section_spacing']
        # Alternate the section's drilling direction. Real programmes are
        # directionally consistent within a section; alternating between them
        # keeps the sampling-direction tensor from being hopeless everywhere.
        base_az = 90.0 if s % 2 == 0 else 270.0

        for h in range(p['holes_per_section']):
            hole = f"SYN-{s:02d}-{h:02d}"
            x = ORIGIN_E + h * p['hole_spacing']
            az0 = base_az + rng.normal(0, 4.0)
            dip0 = p['collar_dip'] + rng.normal(0, 2.0)
            length = p['hole_length'] + rng.normal(0, 15.0)

            collar_rows.append({
                COLS['hole']: hole,
                COLS['easting']: round(x + rng.normal(0, 5.0), 2),
                COLS['northing']: round(y + rng.normal(0, 5.0), 2),
                COLS['elev']: round(COLLAR_Z + rng.normal(0, 3.0), 2),
                COLS['length']: round(length, 1),
                COLS['dip']: round(dip0, 2),
                COLS['azm']: round(az0 % 360.0, 2),
                COLS['property']: 'Synthetic',
                COLS['prospect']: f'Section {s:02d}',
            })

            # Downhole deviation: holes lift and wander. A random walk on top
            # of a systematic trend, so desurvey is genuinely exercised rather
            # than reducing to a straight line.
            depths = np.arange(0.0, length + p['survey_interval'],
                               p['survey_interval'])
            lift = 5.0 * depths / max(length, 1.0)          # dip shallows
            drift = 8.0 * depths / max(length, 1.0)         # azimuth swings
            wobble_d = np.cumsum(rng.normal(0, 0.25, len(depths)))
            wobble_a = np.cumsum(rng.normal(0, 0.40, len(depths)))
            for d, ld, dr, wd, wa in zip(depths, lift, drift, wobble_d, wobble_a):
                survey_rows.append({
                    COLS['hole']: hole,
                    COLS['depth']: round(float(d), 2),
                    COLS['dip']: round(float(dip0 + ld + wd), 2),
                    COLS['azm']: round(float((az0 + dr + wa) % 360.0), 2),
                })

    return pd.DataFrame(collar_rows), pd.DataFrame(survey_rows)


def make_example_deposit(seed: int = 0, **overrides) -> ExampleDeposit:
    """Generate a folded Cu-Au deposit with per-sample ground truth.

    Returns tables in this project's internal schema, so they feed `prepare()`
    directly with no ingest step and no column mapping.
    """
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **overrides}
    rng = np.random.default_rng(seed)

    collars, surveys = _drill_layout(p, rng)
    survey_groups = {k: v for k, v in surveys.groupby(COLS['hole'], sort=False)}

    parts = []
    for _, collar in collars.iterrows():
        hole = collar[COLS['hole']]
        length = float(collar[COLS['length']])

        edges = np.arange(0.0, length, p['sample_length'])
        frm, to = edges[:-1], edges[1:]
        mid = 0.5 * (frm + to)

        # The project's own minimum-curvature integration, so grades are placed
        # at exactly the coordinates the pipeline will later recompute.
        path = build_hole_path(collar, survey_groups[hole], mid)
        xyz = path.set_index(COLS['depth']).loc[mid, ['X', 'Y', 'Z']].to_numpy()

        cu, au = _grades(xyz[:, 0], xyz[:, 1], xyz[:, 2], p, rng)
        poles = fold_normal(xyz[:, 0], xyz[:, 1], p)
        strike, dip = normal_to_strike_dip(poles)

        parts.append(pd.DataFrame({
            COLS['hole']: hole,
            COLS['from']: np.round(frm, 2),
            COLS['to']: np.round(to, 2),
            'Cu_pct': cu,
            'Au_gpt': au,
            'true_pole_x': poles[:, 0],
            'true_pole_y': poles[:, 1],
            'true_pole_z': poles[:, 2],
            'true_strike_deg': strike,
            'true_dip_deg': dip,
        }))

    full = pd.concat(parts, ignore_index=True)

    # No src_index here: desurvey_samples assigns that itself, and a column of
    # the same name would collide with it. The truth rides on `samples` as
    # ordinary columns instead, and is carried through to Dataset.active.
    truth_cols = [COLS['hole'], COLS['from'], COLS['to'],
                  'true_pole_x', 'true_pole_y', 'true_pole_z',
                  'true_strike_deg', 'true_dip_deg']

    return ExampleDeposit(samples=full, collars=collars, surveys=surveys,
                          truth=full[truth_cols].copy(), params=p)


def orientation_error(out: pd.DataFrame, active: pd.DataFrame) -> np.ndarray:
    """Angle in degrees between each estimated pole and the true pole.

    `out` is anything carrying `src_index` and pole_vec_x/y/z -- what both
    `run_estimator` and `scripts/compare_methods.py` return. `active` is the
    `Dataset.active` the run was produced from, which carries the truth columns
    through from the synthetic samples table. Matching on `src_index` is what
    makes this correct under the active-node filter.
    """
    need = {'src_index', 'pole_vec_x', 'pole_vec_y', 'pole_vec_z'}
    missing = need - set(out.columns)
    if missing:
        raise KeyError(f"estimator output is missing {sorted(missing)}")
    if 'true_pole_x' not in active.columns:
        raise KeyError("`active` carries no truth columns -- was it prepared "
                       "from a synthetic deposit?")

    t = active.set_index('src_index')
    o = out.set_index('src_index')
    shared = t.index.intersection(o.index)
    if len(shared) == 0:
        raise ValueError("no src_index overlap between truth and output")

    est = o.loc[shared, ['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy(float)
    true = t.loc[shared, ['true_pole_x', 'true_pole_y',
                          'true_pole_z']].to_numpy(float)
    return angular_difference(est, true)


def example_cfg(**overrides) -> dict:
    """A config that matches the synthetic deposit's elements and scale.

    The shipped defaults score Ag/Pb/Zn, which this deposit does not have, so
    calling `make_cfg()` directly against it raises in `build_mineralization_score`.
    """
    base = dict(score_columns=['Cu_pct', 'Au_gpt'],
                radius_m=120.0, min_neighbors=60, min_holes=3,
                distance_sigma_m=80.0)
    base.update(overrides)
    return make_cfg(**base)
