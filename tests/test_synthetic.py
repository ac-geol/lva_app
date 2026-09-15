"""The synthetic deposit must be trustworthy before anything is concluded from it.

The whole value of this module is that the answer is known, so these tests are
mostly about the answer being *right*: the analytic normal is checked against
numerical differentiation of the surface it claims to be normal to, and against
a plane fitted to the surface itself. If those drift, every number measured
against this deposit is meaningless.
"""
import numpy as np
import pandas as pd
import pytest

from core.geometry import (angular_difference, normal_to_strike_dip,
                           strike_dip_to_normal, unitize)
from core.ingest import load_tables
from core.pipeline import prepare, run_estimator
from core.schema import COLS
from core.synthetic import (DEFAULTS, ORIGIN_E, ORIGIN_N, ExampleDeposit,
                            example_cfg, fold_normal, fold_surface,
                            make_example_deposit, orientation_error)


# --------------------------------------------------------------------------
# The ground truth itself.
# --------------------------------------------------------------------------

def test_fold_normal_matches_numerical_differentiation():
    """The analytic normal must be the true normal of `fold_surface`.

    Everything downstream is measured against this, so it is checked against a
    central difference of the surface rather than against itself.
    """
    p = dict(DEFAULTS)
    rng = np.random.default_rng(0)
    x = ORIGIN_E + rng.uniform(0, 1200, 400)
    y = ORIGIN_N + rng.uniform(0, 800, 400)

    h = 0.01
    dfdx = (fold_surface(x + h, y, p) - fold_surface(x - h, y, p)) / (2 * h)
    dfdy = (fold_surface(x, y + h, p) - fold_surface(x, y - h, p)) / (2 * h)
    numeric = unitize(np.column_stack([-dfdx, -dfdy, np.ones_like(x)]))

    assert np.max(angular_difference(fold_normal(x, y, p), numeric)) < 1e-4


def test_fold_normal_matches_a_plane_fitted_to_the_surface():
    """A plane fitted to nearby points ON the surface has the same pole."""
    p = dict(DEFAULTS)
    rng = np.random.default_rng(1)
    x0, y0 = ORIGIN_E + 317.0, ORIGIN_N + 214.0

    # A patch small enough that curvature is negligible.
    dx = rng.uniform(-4, 4, 3000)
    dy = rng.uniform(-4, 4, 3000)
    x, y = x0 + dx, y0 + dy
    pts = np.column_stack([x, y, fold_surface(x, y, p)])
    pts = pts - pts.mean(axis=0)

    _, _, vh = np.linalg.svd(pts, full_matrices=False)
    fitted = vh[2]                                   # smallest singular vector
    assert angular_difference(fitted, fold_normal(x0, y0, p)[0]) < 0.5


def test_truth_strike_dip_round_trips_to_the_truth_pole():
    dep = make_example_deposit(seed=3)
    t = dep.truth
    back = strike_dip_to_normal(t['true_strike_deg'].to_numpy(),
                                t['true_dip_deg'].to_numpy())
    poles = t[['true_pole_x', 'true_pole_y', 'true_pole_z']].to_numpy()
    # angular_difference goes through arccos, whose precision floor near 0 is
    # ~1e-6 degrees. Anything below 1e-4 is exact for every purpose here.
    assert np.max(angular_difference(back, poles)) < 1e-4


def test_the_fold_actually_folds():
    """A deposit whose orientation barely varies would test nothing."""
    dep = make_example_deposit(seed=0)
    dip = dep.truth['true_dip_deg']
    strike = dep.truth['true_strike_deg']
    assert dip.max() - dip.min() > 25.0
    # Opposite limbs must dip in opposing directions, i.e. two strike modes.
    assert strike.nunique() > 100


# --------------------------------------------------------------------------
# The tables, and the pipeline's ability to consume them.
# --------------------------------------------------------------------------

def test_tables_flow_through_ingest_and_prepare_unchanged():
    dep = make_example_deposit(seed=0)
    cfg = example_cfg()
    samples, collars, surveys = load_tables(dep.samples, dep.collars,
                                            dep.surveys, cfg)
    # Ingest must not silently drop holes or samples from a clean deposit.
    assert len(samples) == len(dep.samples)
    assert len(collars) == len(dep.collars)

    ds = prepare(samples, collars, surveys, cfg)
    assert len(ds.active) > 500
    # The truth has to survive desurvey and scoring to be usable at all.
    for c in ('true_pole_x', 'true_pole_y', 'true_pole_z'):
        assert c in ds.active.columns


def test_desurvey_reproduces_the_coordinates_grades_were_assigned_at():
    """Grades are placed using build_hole_path; the pipeline desurveys again.

    If those two disagree, every sample's grade belongs to a different point in
    space than the estimator thinks, and the truth silently decorrelates.
    """
    dep = make_example_deposit(seed=0)
    cfg = example_cfg()
    ds = prepare(*load_tables(dep.samples, dep.collars, dep.surveys, cfg), cfg)

    p = dep.params
    xyz = ds.active[['mid_x', 'mid_y', 'mid_z']].to_numpy()
    expected = fold_normal(xyz[:, 0], xyz[:, 1], p)
    stored = ds.active[['true_pole_x', 'true_pole_y', 'true_pole_z']].to_numpy()
    assert np.max(angular_difference(expected, stored)) < 1e-6


def test_generation_is_deterministic():
    a = make_example_deposit(seed=7)
    b = make_example_deposit(seed=7)
    pd.testing.assert_frame_equal(a.samples, b.samples)
    pd.testing.assert_frame_equal(a.collars, b.collars)
    assert not a.samples['Cu_pct'].equals(make_example_deposit(seed=8).samples['Cu_pct'])


def test_coordinates_are_obviously_synthetic():
    """Nothing here may be mistakeable for a real survey. See CLAUDE.md section 2."""
    dep = make_example_deposit(seed=0)
    e = dep.collars[COLS['easting']]
    n = dep.collars[COLS['northing']]
    assert e.min() > 9000 and e.max() < 100000
    assert n.min() > 9000 and n.max() < 100000
    assert dep.collars[COLS['hole']].str.startswith('SYN-').all()


def test_grades_are_correlated_and_positive():
    dep = make_example_deposit(seed=0)
    s = dep.samples
    assert (s['Cu_pct'] > 0).all() and (s['Au_gpt'] > 0).all()
    assert 0.4 < s['Cu_pct'].corr(s['Au_gpt']) < 0.98
    # A shell, not a uniform block: the top of the distribution is far above
    # the median, or there is no anomaly to find.
    assert s['Cu_pct'].quantile(0.99) > 8 * s['Cu_pct'].median()


def test_survey_deviation_is_real():
    """A perfectly straight hole would not exercise desurvey at all."""
    dep = make_example_deposit(seed=0)
    g = dep.surveys.groupby(COLS['hole'])
    assert (g[COLS['dip']].max() - g[COLS['dip']].min()).median() > 2.0
    assert (g[COLS['azm']].max() - g[COLS['azm']].min()).median() > 2.0


def test_unknown_parameter_is_rejected():
    with pytest.raises(KeyError):
        make_example_deposit(seed=0, wavelenght=800.0)      # typo


# --------------------------------------------------------------------------
# Recovery: the deposit is solvable, and the error helper is correct.
# --------------------------------------------------------------------------

def test_lsq_gradient_recovers_a_gentle_fold():
    """A locally-planar fold with little noise must be recovered well.

    This is a regression guard on the whole chain -- generation, desurvey,
    scoring, neighbourhood, estimator -- not a claim about method ranking.
    """
    dep = make_example_deposit(seed=0, wavelength=1600.0, cu_noise=0.02,
                               au_noise=0.02, shared_noise=0.02)
    cfg = example_cfg(min_neighbors=25)
    ds = prepare(*load_tables(dep.samples, dep.collars, dep.surveys, cfg), cfg)
    out = run_estimator(ds, cfg, 'lsq_gradient')

    err = orientation_error(out, ds.active)
    assert len(err) > 500
    assert np.median(err) < 15.0


def test_orientation_error_is_zero_against_the_truth_itself():
    """Feeding the truth back in must score 0 -- the helper's own guard."""
    dep = make_example_deposit(seed=0)
    cfg = example_cfg()
    ds = prepare(*load_tables(dep.samples, dep.collars, dep.surveys, cfg), cfg)

    perfect = ds.active[['src_index']].copy()
    perfect[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']] = \
        ds.active[['true_pole_x', 'true_pole_y', 'true_pole_z']].to_numpy()
    assert np.max(orientation_error(perfect, ds.active)) < 1e-4   # arccos floor


def test_orientation_error_rejects_output_without_truth():
    dep = make_example_deposit(seed=0)
    cfg = example_cfg()
    ds = prepare(*load_tables(dep.samples, dep.collars, dep.surveys, cfg), cfg)
    out = run_estimator(ds, cfg, 'lsq_gradient')

    with pytest.raises(KeyError):
        orientation_error(out, ds.active.drop(columns=['true_pole_x']))
