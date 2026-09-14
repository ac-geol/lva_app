"""Write a small file of planes with known orientation, for a Leapfrog check.

Dip and dip azimuth are already trusted: `core/geometry.py` records that the
strike convention was verified against Leapfrog exports, and `core/export.py`
only adds `dip_azimuth = strike + 90`. **Pitch is not.** Its definition is
stated explicitly in `pitch_from_lineation`, but nothing has confirmed that
Leapfrog measures the rake from the same reference in the same direction.

This writes planes whose answer is known by construction, so importing the file
and comparing what Leapfrog draws against the expected column settles it.

    python scripts/make_convention_check.py
    -> out/leapfrog_convention_check.csv

Carries no coordinates from any real property: the points sit on a nominal
100 m grid at a round origin, so the file is safe to share.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.export import dip_azimuth_from_strike, pitch_from_lineation  # noqa: E402
from core.geometry import strike_dip_to_normal  # noqa: E402

OUT = ROOT / 'out' / 'leapfrog_convention_check.csv'

# Spread across the circle and across dip, plus rakes at both ends and the
# middle: a convention that is flipped, or measured from the dip direction
# instead of strike, disagrees somewhere in here.
CASES = [
    # (dip, dip_azimuth, rake)
    (0, 0, 0), (20, 0, 0), (20, 0, 90), (20, 0, 135),
    (45, 90, 0), (45, 90, 45), (45, 90, 90), (45, 90, 150),
    (70, 180, 30), (70, 180, 90), (70, 270, 30), (70, 270, 150),
    (89, 45, 10), (89, 45, 90),
]


def in_plane_lineation(strike_deg, dip_deg, rake_deg) -> np.ndarray:
    """Unit vector in the plane, at `rake_deg` from the strike direction."""
    s = np.radians(np.asarray(strike_deg, dtype=float))
    d = np.radians(np.asarray(dip_deg, dtype=float))
    r = np.radians(np.asarray(rake_deg, dtype=float))

    strike_vec = np.column_stack([np.sin(s), np.cos(s), np.zeros_like(s)])
    down_dip = np.column_stack([np.sin(s + np.pi / 2) * np.cos(d),
                                np.cos(s + np.pi / 2) * np.cos(d),
                                -np.sin(d)])
    return np.cos(r)[:, None] * strike_vec + np.sin(r)[:, None] * down_dip


def main() -> None:
    dip = np.array([c[0] for c in CASES], dtype=float)
    dip_az = np.array([c[1] for c in CASES], dtype=float)
    rake = np.array([c[2] for c in CASES], dtype=float)

    strike = np.mod(dip_az - 90.0, 360.0)
    line = in_plane_lineation(strike, dip, rake)

    # Round-trip through the engine's own conversions, so the file carries what
    # the app would actually export for a plane of this orientation.
    pitch = pitch_from_lineation(strike, dip, line)
    normal = strike_dip_to_normal(strike, dip)

    n = len(CASES)
    table = pd.DataFrame({
        'HoleID': [f'CHECK-{i + 1:02d}' for i in range(n)],
        'From_m': np.zeros(n), 'To_m': np.ones(n),
        'X': 10000.0 + 100.0 * np.arange(n), 'Y': 20000.0, 'Z': 0.0,
        'Dip': np.round(dip, 3),
        'DipAzimuth': np.round(dip_azimuth_from_strike(strike), 3),
        'Pitch': np.round(pitch, 3),
        'expected_dip': dip, 'expected_dip_azimuth': dip_az, 'expected_rake': rake,
        'pole_x': np.round(normal[:, 0], 6),
        'pole_y': np.round(normal[:, 1], 6),
        'pole_z': np.round(normal[:, 2], 6),
    })

    # The engine has to reproduce its own inputs before Leapfrog is asked
    # anything -- a mismatch here is a bug in core, not a convention question.
    assert np.allclose(table['Dip'], dip, atol=1e-6)
    assert np.allclose(np.mod(table['DipAzimuth'] - dip_az + 180, 360) - 180, 0, atol=1e-6)
    assert np.allclose(table['Pitch'], rake, atol=1e-6)

    OUT.parent.mkdir(exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"{OUT.relative_to(ROOT)}  —  {n} planes")
    print("\nInternal round trip: dip, dip azimuth and pitch all reproduce their")
    print("inputs. What remains is whether Leapfrog reads Pitch the same way;")
    print("see docs/LEAPFROG_CHECK.md.")


if __name__ == '__main__':
    main()
