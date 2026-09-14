# Verifying the export conventions against Leapfrog

Two of the three exported angles are already trusted. One is not, and this is
how to settle it.

## What is already verified

`core/geometry.py` states that its strike/dip convention was checked against
Leapfrog exports: strike under the right-hand rule, from an upward pole, with
the dip azimuth at strike + 90. `core/export.py` adds nothing to that beyond
the `+ 90`, so **Dip** and **DipAzimuth** carry no open question.

`tests/test_geometry.py::test_dip_azimuth_is_strike_plus_90` pins that
relationship over the whole circle, and
`test_leapfrog_dip_azimuth_matches_the_exported_pole` pins that the written
angles and the engine's own pole describe the same plane.

## What is not

**Pitch.** The engine has a well-defined in-plane axis for `shape_pca` — the
major eigenvector — and `core/export.py:pitch_from_lineation` measures its rake
from the strike direction, fixing the sense so it plunges downward. That
definition is stated in the docstring precisely so it can be checked. What has
never been confirmed is that **Leapfrog measures pitch from the same reference,
in the same direction.**

The plausible ways it could differ:

- measured from the **dip direction** rather than the strike direction (90° out);
- measured in the **opposite sense** around the plane (pitch vs 180 − pitch);
- measured from the *other* strike end (also 180 − pitch).

None of these show up in any internal metric. A wrong pitch produces an
anisotropy ellipsoid that is correctly oriented as a *plane* but whose long axis
points the wrong way within it — which is exactly the thing LVA angle coding
consumes.

## The check

```bash
python scripts/make_convention_check.py     # -> out/leapfrog_convention_check.csv
```

14 planes spanning dip 0–89°, dip azimuth around the circle, and rakes at 0°,
~45°, 90° and >90°. Each row carries both what the engine exports (`Dip`,
`DipAzimuth`, `Pitch`) and what it should mean (`expected_dip`,
`expected_dip_azimuth`, `expected_rake`), plus the pole vector.

The script asserts the engine reproduces its own inputs before writing, so any
mismatch you then see in Leapfrog is a convention difference, not a bug in
`core`.

Then, in Leapfrog:

1. Import the CSV as planar structural data, mapping **Dip** and **Dip
   Azimuth**. Confirm the discs match `expected_dip` / `expected_dip_azimuth` —
   this should pass, and failing it means something more basic is wrong.
2. Use `Pitch` wherever the ellipsoid orientation takes a third angle. For the
   rows with `expected_rake` 0, the long axis should lie **along strike**; for
   90 it should point **straight down dip**.
3. Row `CHECK-08` (dip 45, dip azimuth 090, rake 150) is the discriminating
   one. Under this engine's definition its long axis plunges shallowly, pointing
   back toward the up-strike side. If Leapfrog draws it plunging the other way,
   the sense is reversed and the fix is `180 - pitch`.

## If it disagrees

Fix it in one place: `core/export.py:pitch_from_lineation`. Do not correct it in
the app or in a spreadsheet — the 3D view and the export both derive from the
engine's pole and lineation, and they must not diverge.

Record the outcome here and in the module docstring, which currently says the
convention is unverified. Until that line is changed, treat exported pitch as
provisional and do not code a grade estimate from it.
