"""The desurveyed-point input path, and the Leapfrog export conventions.

Two things are pinned here. First, that feeding the engine a table of
already-desurveyed points produces *the same field* as feeding it the three raw
tables -- if those ever diverge, the browser app and the CLI stop being the
same tool. Second, the angle conventions the export writes, since a wrong
convention is invisible in every internal metric and only shows up as an
anisotropy ellipsoid pointing the wrong way.
"""
import numpy as np
import pandas as pd
import pytest

from core.export import (LEAPFROG_COLUMNS, dip_azimuth_from_strike,
                         pitch_from_lineation, to_leapfrog)
from core.geometry import angular_difference, strike_dip_to_normal
from core.ingest import load_points, load_tables
from core.pipeline import (hole_paths_from_points, prepare, prepare_points,
                           run_estimator)
from core.schema import COLS
from core.synthetic import example_cfg, make_example_deposit


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------

def _three_table_run(cfg, seed=0):
    dep = make_example_deposit(seed=seed)
    tables = load_tables(dep.samples, dep.collars, dep.surveys, cfg)
    ds = prepare(*tables, cfg)
    return ds, run_estimator(ds, cfg, 'shape_pca')


def _as_point_export(des: pd.DataFrame, score_columns) -> pd.DataFrame:
    """Rebuild what Leapfrog would export from a desurveyed frame.

    Deliberately uses the header names a real export carries -- X / Y / Z --
    rather than the internal ones, so alias resolution is exercised too.
    """
    out = des[[COLS['hole'], COLS['from'], COLS['to'], *score_columns]].copy()
    out['X'] = des['mid_x'].to_numpy()
    out['Y'] = des['mid_y'].to_numpy()
    out['Z'] = des['mid_z'].to_numpy()
    return out.reset_index(drop=True)


def _in_plane_lineation(strike_deg, dip_deg, rake_deg):
    """A unit lineation at a known rake, measured from the strike direction."""
    s = np.radians(np.asarray(strike_deg, dtype=float))
    d = np.radians(np.asarray(dip_deg, dtype=float))
    r = np.radians(np.asarray(rake_deg, dtype=float))

    strike_vec = np.column_stack([np.sin(s), np.cos(s), np.zeros_like(s)])
    down_dip = np.column_stack([np.sin(s + np.pi / 2) * np.cos(d),
                                np.cos(s + np.pi / 2) * np.cos(d),
                                -np.sin(d)])
    return np.cos(r)[:, None] * strike_vec + np.sin(r)[:, None] * down_dip


# --------------------------------------------------------------------------
# The point path must equal the three-table path.
# --------------------------------------------------------------------------

def test_point_path_reproduces_three_table_path():
    """Same deposit, two routes in, one field out.

    This is the whole justification for `prepare_points`: the coordinates are
    already known, so skipping desurvey must change nothing at all.
    """
    cfg = example_cfg()
    ds3, out3 = _three_table_run(cfg)

    pts = load_points(_as_point_export(ds3.des, cfg['score_columns']), cfg)
    dsp = prepare_points(pts, cfg)
    outp = run_estimator(dsp, cfg, 'shape_pca')

    assert len(outp) == len(out3)
    assert np.array_equal(dsp.hole_ids, ds3.hole_ids)
    assert np.allclose(dsp.coords, ds3.coords)

    # Bitwise, not merely close: the two routes hand the estimator the same
    # float64s, so anything less would be hiding a real difference. (Note that
    # `angular_difference` cannot express this -- arccos near 1 gives it a
    # noise floor around 1.7e-6 deg even for identical vectors.)
    poles = ['pole_vec_x', 'pole_vec_y', 'pole_vec_z']
    assert np.array_equal(out3[poles].to_numpy(), outp[poles].to_numpy())
    for col in ('plane_strike_deg', 'plane_dip_deg', 'planarity_norm',
                'local_n', 'local_holes'):
        assert np.allclose(out3[col].to_numpy(float),
                           outp[col].to_numpy(float))


def test_point_path_works_with_a_single_assay_column():
    """The app offers one column, so the one-column case is the shipped case."""
    cfg = example_cfg(score_columns=['Cu_pct'])
    ds3, out3 = _three_table_run(cfg)

    pts = load_points(_as_point_export(ds3.des, cfg['score_columns']), cfg)
    outp = run_estimator(prepare_points(pts, cfg), cfg, 'shape_pca')

    assert len(outp) == len(out3) > 0
    poles = ['pole_vec_x', 'pole_vec_y', 'pole_vec_z']
    assert np.array_equal(out3[poles].to_numpy(), outp[poles].to_numpy())


def test_load_points_requires_coordinates():
    cfg = example_cfg()
    dep = make_example_deposit(seed=0)
    ds = prepare(*load_tables(dep.samples, dep.collars, dep.surveys, cfg), cfg)

    pts = _as_point_export(ds.des, cfg['score_columns']).drop(columns='Z')
    with pytest.raises(KeyError, match='Elevation'):
        load_points(pts, cfg)


def test_load_points_drops_rows_with_no_coordinate():
    """A null coordinate would otherwise reach the neighbourhood graph as NaN."""
    cfg = example_cfg()
    dep = make_example_deposit(seed=0)
    ds = prepare(*load_tables(dep.samples, dep.collars, dep.surveys, cfg), cfg)

    pts = _as_point_export(ds.des, cfg['score_columns'])
    pts.loc[pts.index[:5], 'Y'] = np.nan

    loaded = load_points(pts, cfg)
    assert len(loaded) == len(pts) - 5
    assert loaded[[COLS['easting'], COLS['northing'], COLS['elev']]].notna().all().all()


def test_prepare_points_refuses_compositing():
    """Composited grades with uncomposited coordinates is a silent wrong answer."""
    cfg = example_cfg(composite_length_m=5.0)
    dep = make_example_deposit(seed=0)
    base = example_cfg()
    ds = prepare(*load_tables(dep.samples, dep.collars, dep.surveys, base), base)

    pts = load_points(_as_point_export(ds.des, base['score_columns']), base)
    with pytest.raises(ValueError, match='composite'):
        prepare_points(pts, cfg)


def test_hole_paths_run_downhole_and_cover_every_hole():
    cfg = example_cfg()
    dep = make_example_deposit(seed=0)
    ds = prepare(*load_tables(dep.samples, dep.collars, dep.surveys, cfg), cfg)

    pts = load_points(_as_point_export(ds.des, cfg['score_columns']), cfg)
    dsp = prepare_points(pts, cfg)

    assert set(dsp.paths) == set(pts[COLS['hole']].astype(str))
    for path in dsp.paths.values():
        depths = path[COLS['depth']].to_numpy()
        assert len(path) > 1
        assert np.all(np.diff(depths) > 0)
        assert path[['X', 'Y', 'Z']].notna().all().all()


# --------------------------------------------------------------------------
# Leapfrog angle conventions.
# --------------------------------------------------------------------------

def test_dip_azimuth_wraps_past_north():
    assert dip_azimuth_from_strike([0.0]) == pytest.approx([90.0])
    assert dip_azimuth_from_strike([300.0]) == pytest.approx([30.0])
    assert dip_azimuth_from_strike([270.0]) == pytest.approx([0.0])
    assert np.all((dip_azimuth_from_strike(np.arange(0, 360, 7.0)) >= 0)
                  & (dip_azimuth_from_strike(np.arange(0, 360, 7.0)) < 360))


def test_pitch_recovers_a_known_rake():
    """Build a lineation at a known rake, and read the rake back."""
    strikes = np.repeat(np.arange(0.0, 360.0, 30.0), 4)
    dips = np.tile([15.0, 40.0, 65.0, 85.0], 12)

    for rake in (5.0, 35.0, 90.0, 140.0, 175.0):
        rakes = np.full_like(strikes, rake)
        line = _in_plane_lineation(strikes, dips, rakes)
        assert pitch_from_lineation(strikes, dips, line) == \
            pytest.approx(rakes, abs=1e-6)


def test_pitch_is_insensitive_to_lineation_sign():
    """Eigenvectors carry an arbitrary sign; pitch must not inherit it."""
    strikes = np.arange(0.0, 360.0, 45.0)
    dips = np.full_like(strikes, 50.0)
    line = _in_plane_lineation(strikes, dips, np.full_like(strikes, 40.0))

    assert pitch_from_lineation(strikes, dips, line) == \
        pytest.approx(pitch_from_lineation(strikes, dips, -line), abs=1e-9)


def test_pitch_of_a_down_dip_lineation_is_90():
    strikes = np.arange(0.0, 360.0, 45.0)
    dips = np.full_like(strikes, 55.0)
    line = _in_plane_lineation(strikes, dips, np.full_like(strikes, 90.0))
    assert pitch_from_lineation(strikes, dips, line) == \
        pytest.approx(np.full_like(strikes, 90.0), abs=1e-6)


def test_pitch_ignores_any_component_off_the_plane():
    """Only the in-plane part of the lineation can matter."""
    strikes = np.arange(0.0, 360.0, 60.0)
    dips = np.full_like(strikes, 35.0)
    line = _in_plane_lineation(strikes, dips, np.full_like(strikes, 25.0))
    normal = strike_dip_to_normal(strikes, dips)

    contaminated = line + 0.75 * normal
    assert pitch_from_lineation(strikes, dips, contaminated) == \
        pytest.approx(pitch_from_lineation(strikes, dips, line), abs=1e-6)


# --------------------------------------------------------------------------
# The exported file.
# --------------------------------------------------------------------------

def test_leapfrog_export_schema_and_ranges():
    cfg = example_cfg()
    _, out = _three_table_run(cfg)
    lf = to_leapfrog(out, vs_reference_deg=np.full(len(out), 12.5))

    assert list(lf.columns) == LEAPFROG_COLUMNS
    assert len(lf) == len(out)
    assert np.all((lf['Dip'] >= 0) & (lf['Dip'] <= 90))
    assert np.all((lf['DipAzimuth'] >= 0) & (lf['DipAzimuth'] < 360))
    assert np.all((lf['Pitch'] >= 0) & (lf['Pitch'] <= 180))
    assert lf.notna().all().all()
    assert (lf['Estimator'] == 'shape_pca').all()


def test_leapfrog_dip_azimuth_matches_the_exported_pole():
    """The written angles and the engine's pole must describe one plane."""
    cfg = example_cfg()
    _, out = _three_table_run(cfg)
    lf = to_leapfrog(out)

    rebuilt = strike_dip_to_normal(
        (lf['DipAzimuth'].to_numpy() - 90.0) % 360.0, lf['Dip'].to_numpy())
    poles = out[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
    assert np.max(angular_difference(rebuilt, poles)) < 1e-3


def test_pitch_is_suppressed_where_there_is_no_in_plane_axis():
    """lsq_gradient reads ~1e-17 linearity; its pitch must not be reported."""
    cfg = example_cfg()
    _, out = _three_table_run(cfg)

    out = out.copy()
    out['linearity_norm'] = 0.0
    assert (to_leapfrog(out)['Pitch'] == 0.0).all()

    out['linearity_norm'] = 0.5
    assert (to_leapfrog(out)['Pitch'] > 0.0).any()


def test_vs_reference_column_is_empty_when_not_supplied():
    cfg = example_cfg()
    _, out = _three_table_run(cfg)
    assert to_leapfrog(out)['VsReferenceDeg'].isna().all()
