"""Desurvey against hand-computable cases and the original loop form."""
import numpy as np
import pandas as pd
import pytest

from core.desurvey import build_hole_path, dip_az_to_vector, normalize_rows
from core.schema import COLS


def _collar(x=1000.0, y=2000.0, z=500.0, dip=-90.0, az=0.0, length=300.0):
    return pd.Series({COLS['easting']: x, COLS['northing']: y, COLS['elev']: z,
                      COLS['dip']: dip, COLS['azm']: az, COLS['length']: length})


def _survey(rows):
    return pd.DataFrame(rows, columns=[COLS['depth'], COLS['dip'], COLS['azm']])


def test_dip_az_to_vector_straight_down():
    v = dip_az_to_vector([-90.0], [0.0])[0]
    assert v == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def test_dip_az_to_vector_horizontal_east():
    v = dip_az_to_vector([0.0], [90.0])[0]
    assert v == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


def test_vertical_hole_goes_straight_down():
    path = build_hole_path(_collar(), _survey([[0.0, -90.0, 0.0]]),
                           np.array([0.0, 100.0, 200.0]))
    at200 = path.loc[path[COLS['depth']] == 200.0].iloc[0]
    assert (at200['X'], at200['Y']) == pytest.approx((1000.0, 2000.0), abs=1e-9)
    assert at200['Z'] == pytest.approx(300.0)          # 500 - 200


def test_inclined_straight_hole_matches_trigonometry():
    dip, az, depth = -60.0, 45.0, 100.0
    path = build_hole_path(_collar(dip=dip, az=az),
                           _survey([[0.0, dip, az]]), np.array([0.0, depth]))
    end = path.loc[path[COLS['depth']] == depth].iloc[0]
    horiz = depth * np.cos(np.radians(abs(dip)))
    assert end['X'] - 1000.0 == pytest.approx(horiz * np.sin(np.radians(az)))
    assert end['Y'] - 2000.0 == pytest.approx(horiz * np.cos(np.radians(az)))
    assert end['Z'] - 500.0 == pytest.approx(-depth * np.sin(np.radians(abs(dip))))


def test_survey_not_starting_at_zero_seeds_from_the_collar():
    """A survey whose first station is below the collar must not lose that gap."""
    path = build_hole_path(_collar(dip=-90.0, az=0.0),
                           _survey([[50.0, -90.0, 0.0]]),
                           np.array([0.0, 25.0, 50.0]))
    assert path[COLS['depth']].min() == 0.0
    at25 = path.loc[path[COLS['depth']] == 25.0].iloc[0]
    assert at25['Z'] == pytest.approx(475.0)


def test_path_is_extended_past_the_last_survey_station():
    path = build_hole_path(_collar(), _survey([[0.0, -90.0, 0.0]]),
                           np.array([0.0, 400.0]))
    assert path[COLS['depth']].max() >= 400.0


def test_deviated_hole_matches_the_original_loop_implementation():
    """Vectorized minimum curvature == the notebook's step-by-step integration."""
    rng = np.random.default_rng(7)
    md = np.sort(rng.uniform(0, 400, 25))
    md[0] = 0.0
    sv = _survey(np.column_stack([
        md, -80.0 + np.cumsum(rng.normal(0, 1.5, len(md))),
        30.0 + np.cumsum(rng.normal(0, 2.0, len(md)))]))
    targets = np.sort(rng.uniform(0, 400, 60))
    collar = _collar(dip=float(sv[COLS['dip']].iloc[0]),
                     az=float(sv[COLS['azm']].iloc[0]))

    path = build_hole_path(collar, sv, targets)

    # Reference: the explicit per-segment loop the notebook used.
    depths = path[COLS['depth']].to_numpy()
    dirs = normalize_rows(path[['ux', 'uy', 'uz']].to_numpy())
    ref = np.empty((len(depths), 3))
    ref[0] = [1000.0, 2000.0, 500.0]
    for i in range(1, len(depths)):
        seg = depths[i] - depths[i - 1]
        u1, u2 = dirs[i - 1], dirs[i]
        dogleg = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
        rf = 1.0 if dogleg < 1e-12 else (2.0 / dogleg) * np.tan(dogleg / 2.0)
        ref[i] = ref[i - 1] + 0.5 * seg * (u1 + u2) * rf

    assert path[['X', 'Y', 'Z']].to_numpy() == pytest.approx(ref, abs=1e-9)
