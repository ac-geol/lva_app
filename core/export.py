"""Orientation field -> the angle conventions Leapfrog expects.

Leapfrog takes planar structural data as DIP and DIP AZIMUTH, and its
variogram / search ellipsoid takes DIP, DIP AZIMUTH and PITCH. The engine works
in poles and eigenvectors, so this module is the single place that conversion
happens -- the same role `estimators.field_to_orientations` plays for the
internal angles.

No file I/O here, in keeping with the rest of `core/`: these functions return a
DataFrame and the caller decides where it goes.

Convention status
-----------------
DIP and DIP AZIMUTH are safe. `geometry.normal_to_strike_dip` produces strike
under the right-hand rule with dip azimuth at strike + 90, and that
relationship is recorded in that module as verified against Leapfrog exports.

PITCH IS NOT YET VERIFIED against a Leapfrog round trip. The definition used
here is stated explicitly in `pitch_from_lineation` so that a round trip can
confirm or correct it. Do not code a grade estimate from exported pitch before
that check has been done.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import strike_dip_to_normal, unitize
from .schema import COLS

# The exported schema, in order. Stable: downstream imports and any saved
# Leapfrog mapping depend on these names not moving.
LEAPFROG_COLUMNS = [
    'HoleID', 'From_m', 'To_m', 'X', 'Y', 'Z',
    'Dip', 'DipAzimuth', 'Pitch',
    'Planarity', 'Linearity', 'LocalN', 'LocalHoles',
    'VsReferenceDeg', 'Estimator',
]

# Below this normalized linearity there is no in-plane axis worth reporting and
# pitch is noise. lsq_gradient builds an essentially rank-one tensor and reads
# ~1e-17 here; shape_pca returned 0.33-0.61 across the test property.
MIN_LINEARITY_FOR_PITCH = 0.05


def dip_azimuth_from_strike(strike_deg) -> np.ndarray:
    """Right-hand-rule strike -> true down-dip direction, 0-360."""
    return np.mod(np.asarray(strike_deg, dtype=float) + 90.0, 360.0)


def pitch_from_lineation(strike_deg, dip_deg, line_vec, *,
                         eps: float = 1e-9) -> np.ndarray:
    """Rake of the in-plane axis, measured from the strike direction.

    Definition applied here, stated so a Leapfrog round trip can check it:

      * the plane is taken from (strike, dip), and its upward pole from
        `geometry.strike_dip_to_normal`;
      * the strike direction is the horizontal unit vector at `strike_deg`;
      * the lineation is projected into the plane, and its sense is fixed so
        that it plunges downward -- the estimator's eigenvector has arbitrary
        sign, so without this the angle would flip between neighbouring nodes;
      * pitch is the angle between those two in-plane directions, 0-180.

    A lineation lying exactly horizontal in the plane is sign-ambiguous under
    that rule and resolves to the unflipped branch; the alternative reading is
    180 - pitch. A plane at dip 0 has a degenerate strike, inherited from
    `normal_to_strike_dip`, and its pitch is correspondingly arbitrary --
    which is harmless, because an isotropic-in-plane ellipsoid is the right
    answer there anyway.
    """
    strike = np.radians(np.asarray(strike_deg, dtype=float))
    normal = strike_dip_to_normal(strike_deg, dip_deg)
    strike_vec = np.column_stack(
        [np.sin(strike), np.cos(strike), np.zeros_like(strike)])

    line = unitize(np.atleast_2d(np.asarray(line_vec, dtype=float)))
    in_plane = line - np.einsum('ni,ni->n', line, normal)[:, None] * normal
    norm = np.linalg.norm(in_plane, axis=1)
    in_plane = in_plane / np.where(norm < eps, 1.0, norm)[:, None]

    # Axial data has no direction; force the down-plunge sense so the reported
    # angle is single-valued across the field.
    in_plane = np.where(in_plane[:, 2:3] > 0, -in_plane, in_plane)

    cos = np.clip(np.einsum('ni,ni->n', strike_vec, in_plane), -1.0, 1.0)
    return np.where(norm < eps, 0.0, np.degrees(np.arccos(cos)))


def to_leapfrog(orientations: pd.DataFrame, *, vs_reference_deg=None,
                min_linearity: float = MIN_LINEARITY_FOR_PITCH,
                decimals: int = 3) -> pd.DataFrame:
    """Convert a `run_estimator` result into the Leapfrog import schema.

    `vs_reference_deg` is the per-node angle to the drill-pattern reference.
    It is carried into the file rather than left in the app so that a decision
    made from this export can still be audited against the drill pattern
    months later. None leaves the column empty.

    Pitch is emitted only where the node has a real in-plane axis; below
    `min_linearity` it falls back to 0.0 rather than reporting a confident
    angle derived from numerical noise.
    """
    strike = orientations['plane_strike_deg'].to_numpy(float)
    dip = orientations['plane_dip_deg'].to_numpy(float)
    line = orientations[['line_vec_x', 'line_vec_y', 'line_vec_z']].to_numpy(float)

    linearity = orientations['linearity_norm'].to_numpy(float)
    pitch = pitch_from_lineation(strike, dip, line)
    pitch = np.where(np.nan_to_num(linearity) >= min_linearity, pitch, 0.0)

    n = len(orientations)
    out = pd.DataFrame({
        'HoleID': orientations[COLS['hole']].astype(str).to_numpy(),
        'From_m': orientations[COLS['from']].to_numpy(float),
        'To_m': orientations[COLS['to']].to_numpy(float),
        'X': orientations['mid_x'].to_numpy(float),
        'Y': orientations['mid_y'].to_numpy(float),
        'Z': orientations['mid_z'].to_numpy(float),
        'Dip': np.round(dip, decimals),
        'DipAzimuth': np.round(dip_azimuth_from_strike(strike), decimals),
        'Pitch': np.round(pitch, decimals),
        'Planarity': np.round(orientations['planarity_norm'].to_numpy(float), 4),
        'Linearity': np.round(linearity, 4),
        'LocalN': orientations['local_n'].to_numpy(),
        'LocalHoles': orientations['local_holes'].to_numpy(),
        'VsReferenceDeg': (np.full(n, np.nan) if vs_reference_deg is None
                           else np.round(np.asarray(vs_reference_deg, float),
                                         decimals)),
        'Estimator': orientations['estimator'].to_numpy(),
    })
    return out[LEAPFROG_COLUMNS]
