"""Angle conventions, checked against hand-worked geometry."""
import numpy as np
import pytest

from core.geometry import (angular_difference, normal_to_strike_dip,
                           orientation_tensor, strike_dip_to_normal, unitize,
                           vector_to_trend_plunge, woodcock)


def _circ_close(a, b, tol=1e-6):
    """Compare azimuths modulo 360 so 0 and 360 are the same bearing."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return bool(np.all(np.abs((a - b + 180.0) % 360.0 - 180.0) < tol))


def test_horizontal_plane_has_zero_dip():
    strike, dip = normal_to_strike_dip(np.array([0.0, 0.0, 1.0]))
    assert dip == pytest.approx(0.0, abs=1e-9)


def test_plane_dipping_30_east():
    # A plane dipping 30 deg toward azimuth 090 has upward normal leaning east.
    n = np.array([np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))])
    strike, dip = normal_to_strike_dip(n)
    assert dip == pytest.approx(30.0)
    assert strike == pytest.approx(0.0)          # dip azimuth 090 = strike + 90


def test_dip_azimuth_is_strike_plus_90():
    # The relationship the Leapfrog export depends on, over the whole circle.
    for true_strike in range(0, 360, 15):
        for true_dip in (5, 30, 60, 85):
            n = strike_dip_to_normal(true_strike, true_dip)
            strike, dip = normal_to_strike_dip(n)
            assert dip == pytest.approx(true_dip, abs=1e-6)
            assert _circ_close(strike, true_strike)
            assert _circ_close(strike + 90,
                               np.degrees(np.arctan2(n[0, 0], n[0, 1])))


def test_pole_sign_does_not_change_the_plane():
    n = unitize(np.array([0.3, -0.7, 0.65]))
    assert normal_to_strike_dip(n) == pytest.approx(normal_to_strike_dip(-n))


def test_trend_plunge_of_straight_down():
    trend, plunge = vector_to_trend_plunge(np.array([0.0, 0.0, -1.0]))
    assert plunge == pytest.approx(90.0)


def test_trend_plunge_horizontal_north():
    trend, plunge = vector_to_trend_plunge(np.array([0.0, 1.0, 0.0]))
    assert plunge == pytest.approx(0.0, abs=1e-9)
    assert trend % 360 == pytest.approx(0.0, abs=1e-9)


def test_trend_plunge_shallow_northeast():
    v = np.array([1.0, 1.0, -np.tan(np.radians(20)) * np.sqrt(2)])
    trend, plunge = vector_to_trend_plunge(v)
    assert trend == pytest.approx(45.0)
    assert plunge == pytest.approx(20.0)


def test_vectorized_matches_scalar():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    t, p = vector_to_trend_plunge(v)
    for k in range(50):
        tk, pk = vector_to_trend_plunge(v[k])
        assert (tk, pk) == pytest.approx((t[k], p[k]))


def test_woodcock_tight_cluster_vs_uniform():
    rng = np.random.default_rng(1)
    axis = unitize(np.array([0.2, 0.3, 0.93]))
    tight = unitize(axis + 0.02 * rng.normal(size=(500, 3)))
    uniform = unitize(rng.normal(size=(500, 3)))

    w_tight, w_uniform = woodcock(tight), woodcock(uniform)
    assert w_tight['S1'] > 0.99
    assert w_uniform['S1'] < 0.45          # 1/3 plus sampling noise
    assert w_tight['C'] > w_uniform['C']
    assert angular_difference(w_tight['v1'], axis)[0] < 2.0


def test_orientation_tensor_is_sign_invariant():
    rng = np.random.default_rng(2)
    v = unitize(rng.normal(size=(200, 3)))
    flipped = v * rng.choice([-1.0, 1.0], size=(200, 1))
    a, _ = orientation_tensor(v)
    b, _ = orientation_tensor(flipped)
    assert a == pytest.approx(b)
