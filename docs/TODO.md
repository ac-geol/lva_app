# Next session

## 1. Standing rule: no data in git

**Nothing containing potentially sensitive data gets committed or pushed on
this project.** Local data only.

This covers derived artifacts, not just raw assay tables — anything carrying
coordinates, hole IDs, or grades. Check what is being staged before `git add
-A`; that command is how data gets in.

`.gitignore` is deny-by-default: `*.csv` and `data/` are ignored outright, with
explicit `!` exceptions for the aggregate-only tables (`method_comparison_summary`,
`lsq_tuning`, `contact_validation` — angles, counts and timings, no coordinates,
no hole IDs). A new data file is never staged by accident.

## 2. The repo — done, and staying private

The remote was **deleted and recreated data-free on 2026-09-03**, then pushed
with a single clean root commit. Verified afterwards: one branch, 0 forks,
visibility PRIVATE, and every old SHA — including the root that held the four
MPA CSVs — returns "No commit found". A force-push would not have been enough;
GitHub keeps unreachable commits retrievable by direct SHA.

`feature/downhole-compositing` was not recreated. Its content is in `main`, and
its history descended from the data-bearing root.

**Going public is now purely a decision, not a blocked one.** It is deliberately
not being taken yet. If it ever is, `docs/SETUP.md` §2 documents the private
clone-with-auth path and would need updating.

Rollback of the pre-scrub history exists locally at
`~/lva_app_pre_scrub_history.bundle`. It **contains the MPA data and the
notebook** — local rollback only, never to be copied or pushed.

---

## 3. PARKED: shape_pca beats lsq_gradient on synthetic ground truth

**Status: parked deliberately on 2026-09-03. Do not act on this without a
decision.**

`core/synthetic.py` generates a folded deposit where the true orientation is
known analytically at every sample. On it:

| | true error |
|---|---|
| `shape_pca` | **15.6°** |
| `lsq_gradient` | 21.9° |
| drill-pattern reference | 15.8° |

This is the same ordering MacPass gave (21.5 / 29.2 / 21.9), but here the truth
is independent of any interpretation, so the circularity argument that
justified preferring `lsq_gradient` does not apply.

**The gap is in the argument, not the code.** `METHOD_COMPARISON.md` §2 reasons:
*shape_pca ≈ control → it is measuring drilling → circular → useless for the
goal.* But the control is **not grade-free**. `active_score_quantile: 0.85`
selects the node cloud *by grade* before any estimator runs; the control only
strips grade from the edge **weights**. Both see a grade-defined cloud. Where
grade defines a sheet, the shape of that cloud is the geology — not the drill
pattern.

Held across wavelength 400–1600 m, shell thickness 25–300 m, and hole spacing
80–250 m. `lsq_gradient` did not win in any configuration tried.

**Caveat, and the untested case.** The synthetic fold is favourable to PCA:
continuous, unfaulted, well bracketed in 3D. The drill sections *alternate*
azimuth. The sharpest untried test is a **single-azimuth programme**, which is
the geometry most likely to break PCA — that is the experiment to run first if
this is picked up.

**If PCA keeps winning, it may be the only method needed**, and the docs' central
claim needs rewriting to what the evidence supports: shape_pca is *redundant with
geometry*, not *wrong*.

---

## 4. FOR REVIEW: is `min_neighbor_length_m` actually needed?

**Not implemented. The case for it was not accepted, and this is the evidence
offered for review — not a decision.**

The concern: `core/estimators.py:81` gates every node on
`edges.neighbor_counts() >= cfg['min_neighbors']` (default 100), an absolute
sample count. Composite upstream and the same number means something different.

Run on the synthetic deposit (wavelength 1600 m, low noise), compositing to
several lengths with the count held at 100:

```
FIXED COUNT  min_neighbors = 100 throughout
 L (m)  active  valid  holes  x-extent  true err
  None   3,349  2,219     61       952      7.42
   2.0   2,514  1,316     50       871      7.37
   3.0   1,679    136     15       688      7.41
   5.0   1,007      0      0         0       nan
   8.0     629      0      0         0       nan
```

**The point is not that the error gets worse. It is that it does not.** At
L = 3 the field has lost 94% of its nodes and 75% of its holes, and the error
against known truth still reads **7.41° — indistinguishable from the
uncomposited 7.42°**. Every quality number you would think to check says the
run is fine. Two steps further it silently returns nothing at all, with no
message saying why.

Rescaling the count to hold roughly the same metres of core removes it
entirely — same coverage, same accuracy, all 61 holes retained at every length:

```
METRES-EQUIVALENT  min_neighbors rescaled to hold ~150 m of core
 L (m)  min_n  valid  holes  x-extent  true err
  None    100  2,219     61       952      7.42
   2.0     75  1,649     61       951      7.82
   3.0     50  1,112     61       952      7.59
   5.0     30    669     61       949      8.05
   8.0     19    412     61       949      7.19
```

Reproduce both with `core/synthetic.py` — truth is analytic, so it can be
re-derived at composited midpoints even though compositing drops the columns.

**Two ways to fix it, and the cheaper one may be enough:**

1. **A warning, not a new parameter.** Print the effective metres of core behind
   `min_neighbors` at run time (`min_neighbors x median support`), and warn when
   support changes. Leaves the config alone; makes the silent failure loud.
2. **A `min_neighbor_length_m` gate** (~127 m for MacPass, ~150 m here) replacing
   the count. Self-calibrating, but adds a config concept.

Option 1 is the smaller change and addresses the actual failure, which is
silence rather than strictness. That is the recommendation.

---

## 5. Smaller open items

- **Scripts hardcode the MacPass filenames** — `scripts/compare_methods.py`,
  `validate_contacts.py`, `tune_lsq.py`. They need a `--data-dir` flag.
  `--synthetic` already removes the need for any data at all, so a fresh clone
  can produce a stereonet; this is now about convenience, not access.
- **`README.md` and `docs/METHOD_COMPARISON.md` cite MacPass numbers throughout.**
  Both say explicitly that the numbers are a written record and not reproducible
  from a clean clone. They stay valid as reasoning.
- **Compositing is not this project's job** — see `docs/COMPOSITING.md`.
  `core/compositing.py` stays as a reference implementation, off by default.
