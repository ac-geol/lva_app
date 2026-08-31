"""Vector <-> geological angle conversions, and orientation-tensor statistics.

Angle conventions (verified against the notebook's Leapfrog exports):
  trend/plunge -- the vector is flipped to point downward, trend is the
      azimuth of its horizontal projection, plunge is positive downward.
  strike/dip   -- from an *upward* pole. dip_azimuth = strike + 90, i.e. the
      true dip direction; strike follows the right-hand rule.
"""
from __future__ import annotations

import numpy as np


def _as_rows(v: np.ndarray):
    v = np.asarray(v, dtype=float)
    single = v.ndim == 1
    return np.atleast_2d(v), single


def unitize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def vector_to_trend_plunge(v: np.ndarray):
    """-> (trend_deg, plunge_deg), plunge positive downward."""
    a, single = _as_rows(v)
    a = unitize(a)
    a = np.where(a[:, 2:3] > 0, -a, a)          # point downward
    trend = np.degrees(np.arctan2(a[:, 0], a[:, 1])) % 360.0
    plunge = np.degrees(np.arcsin(np.clip(-a[:, 2], -1.0, 1.0)))
    return (trend[0], plunge[0]) if single else (trend, plunge)


def normal_to_strike_dip(n: np.ndarray):
    """Pole -> (strike_deg, dip_deg) with the right-hand rule."""
    a, single = _as_rows(n)
    a = unitize(a)
    a = np.where(a[:, 2:3] < 0, -a, a)          # upward normal
    dip = np.degrees(np.arccos(np.clip(np.abs(a[:, 2]), -1.0, 1.0)))
    strike = (np.degrees(np.arctan2(a[:, 0], a[:, 1])) - 90.0) % 360.0
    return (strike[0], dip[0]) if single else (strike, dip)


def strike_dip_to_normal(strike_deg, dip_deg) -> np.ndarray:
    """Inverse of normal_to_strike_dip; returns the upward pole."""
    s = np.radians(np.asarray(strike_deg, dtype=float))
    d = np.radians(np.asarray(dip_deg, dtype=float))
    return unitize(np.column_stack([
        np.sin(d) * np.sin(s + np.pi / 2),
        np.sin(d) * np.cos(s + np.pi / 2),
        np.cos(d) * np.ones_like(s),
    ]))


def orientation_tensor(vectors: np.ndarray, weights=None):
    """Normalized orientation tensor of axial data -> (eigvals desc, eigvecs).

    Sign-invariant, so it is the correct summary for undirected poles/lines.
    """
    v = unitize(np.atleast_2d(np.asarray(vectors, dtype=float)))
    w = (np.ones(len(v)) if weights is None
         else np.asarray(weights, dtype=float))
    w = w / w.sum()
    T = np.einsum('n,ni,nj->ij', w, v, v)
    evals, evecs = np.linalg.eigh(T)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def woodcock(vectors: np.ndarray, weights=None) -> dict:
    """Woodcock shape/strength parameters for a set of axes.

    K = ln(S1/S2) / ln(S2/S3):  K > 1 clustered, K < 1 girdle.
    C = ln(S1/S3):              overall strength of preferred orientation.
    S1 is also reported directly -- for poles, S1 -> 1 means one tight cluster,
    i.e. a single well-defined plane across the dataset.
    """
    evals, evecs = orientation_tensor(vectors, weights)
    s1, s2, s3 = np.clip(evals, 1e-12, None)
    k = np.log(s1 / s2) / np.log(s2 / s3) if s2 > s3 and s1 > s2 else np.inf
    return {
        'S1': float(s1), 'S2': float(s2), 'S3': float(s3),
        'K': float(k), 'C': float(np.log(s1 / s3)),
        'v1': evecs[:, 0], 'v2': evecs[:, 1], 'v3': evecs[:, 2],
        'n': int(len(np.atleast_2d(vectors))),
    }


def angular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Acute angle (degrees) between axes -- direction-insensitive."""
    a, b = unitize(np.atleast_2d(a)), unitize(np.atleast_2d(b))
    dot = np.abs(np.einsum('ni,ni->n', a, b))
    return np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))
