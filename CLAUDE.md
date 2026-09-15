# Project context

What this project is for, what has been decided, and what is still open. The
reasoning lives here so the other documents can stay short:
[`README.md`](README.md) says what the repo is,
[`docs/APP_SETUP.md`](docs/APP_SETUP.md) is the user guide, and
[`docs/LEAPFROG_CHECK.md`](docs/LEAPFROG_CHECK.md) is one outstanding
verification procedure.

---

## 1. The objective

Derive local structural orientation — strike and dip — from drillhole assay
data alone. No oriented core, no televiewer.

The use is **LVA angle coding for grade estimation**: giving a search ellipsoid
a local orientation that follows the mineralisation instead of one global
direction. That is a lower bar than picking veins, and it sets what "good
enough" means.

**Accuracy is settled, not blocking.** The goal is discerning major trends, and
a median error around 27° against logged geology is accepted as adequate for
angle coding. Note the tension it creates: angle coding wants a smooth,
continuous field, but §4 shows apparent smoothness and external truth move in
opposite directions here. Any "smooth it for coding" step needs the contact
check, not eyeballing.

## 2. Standing rule: no data in git

**Nothing carrying coordinates, hole IDs or grades is committed or pushed.**
Raw tables and derived files alike. Local data only.

`.gitignore` is deny-by-default: every `*.csv` and `data/` is ignored outright,
so a data file is never staged by accident. Check what is being staged before
`git add -A` — that command is how data gets in.

**The repo is public** (since 2026-09-14). It was deleted and recreated
data-free on 2026-09-03 and pushed as a single clean root commit; every old SHA
returns "No commit found", including the root that once held raw assay tables.
Verified again before going public: one root commit, and the only CSVs ever
committed are aggregate tables of angles, counts and timings. The rule above
matters more now, not less.

A rollback bundle of the pre-scrub history exists locally at
`~/lva_app_pre_scrub_history.bundle`. It **contains real drillhole data** —
local rollback only, never to be copied or pushed.

## 3. What ships, and why

Two methods, answering different questions:

- **`lsq_gradient`** — least-squares reconstruction of the local grade
  gradient. Minimises `Σ w (δ_ij − ∇g·û_ij)²`, whose normal equations contain
  the sampling-direction tensor explicitly and therefore invert it away. That
  is what removes the drill geometry, and it is why solving beats averaging.
- **`shape_pca`** — principal component analysis on the positions of the
  high-grade samples around each point. Ships as the **null**, not as a
  competitor: on the test property its answer sat 2.0° from the pure
  drill-pattern reference. It is largely measuring where the holes went.

Every run also computes a **drill-pattern reference** — the same shape-PCA with
grade stripped out of the edge weights — and reports the angle between it and
the answer. A small angle means the result is describing the drilling rather
than the rock. That number is the check, and the app puts it first.

Three candidates were tried and removed. The graph structure tensor scored
worse than random against logged contacts while reporting high confidence: it
averages over whatever edge directions the drill programme happened to provide,
which estimates a gradient only when those directions are isotropic, and they
are not. The finding that solving beats averaging is pinned by a test that
builds the averaged tensor inline, so deleting the code did not delete the
evidence.

## 4. Two ways to fool yourself, both confirmed here

**Pole concentration is the wrong selection metric.** A tight stereonet rewards
the artifact: the drill programme is globally consistent, so measuring it
produces exactly the clustering that looks like a good answer.

**Smoothing manufactures structure.** Across a tuning sweep, added score
smoothing improved every internal metric while monotonically degrading the
external one. At five passes the field reported confidence 0.92 and spatial
coherence 1.3° while sitting 72.6° from logged geology — worse than random.
Internal confidence and external truth move in opposite directions.

Selection uses split-half stability and the contact check against logged
geology. Both are cheap, both are implemented, and the contact check is the
only metric that can falsify the field.

## 5. Parked: `shape_pca` may actually beat `lsq_gradient`

**Parked deliberately on 2026-09-03. Do not act on this without a decision.**

On a synthetic folded deposit where the true orientation is known analytically
at every sample, `shape_pca` reaches 15.6° true error against `lsq_gradient`'s
21.9° — the same ordering the real data gave, but here the truth is independent
of any interpretation, so the circularity argument that justified preferring
`lsq_gradient` does not apply to it.

The gap is in the argument, not the code. The reasoning runs *shape_pca ≈
control → measuring drilling → circular → useless*, but the control is **not
grade-free**: `active_score_quantile` selects the node cloud by grade before any
estimator runs, and the control only strips grade from the edge weights. Where
grade defines a sheet, the shape of that cloud is the geology.

Held across wavelength 400–1600 m, shell thickness 25–300 m and hole spacing
80–250 m; `lsq_gradient` did not win in any configuration tried. **Caveat:** the
synthetic fold is favourable to PCA — continuous, unfaulted, well bracketed,
with sections that alternate azimuth. The sharpest untried test is a
**single-azimuth programme**. If PCA keeps winning, the central claim needs
rewriting to what the evidence supports: shape_pca is *redundant with geometry*,
not *wrong*.

## 6. `min_neighbors` counts samples, not metres

`min_neighbors` (default 100) gates every node on an absolute sample count. Feed
the engine composites of a different length and the same number means a
different amount of rock — and the failure is silent. On the synthetic deposit,
holding the count at 100 while lengthening the support to 3 m lost 94% of the
nodes and 75% of the holes while the error against known truth stayed at 7.41°,
indistinguishable from the uncomposited 7.42°. Every quality number you would
think to check said the run was fine. Two steps further it returned nothing at
all, with no message saying why.

**The decision was a warning, not a new parameter.** A `min_neighbor_length_m`
gate would be self-calibrating but adds a config concept, and the real failure
is silence rather than strictness. The app therefore shows the median interval
and what the threshold works out to in metres of core, and says so when few
nodes survive.

## 7. Design constraints

`core/` is pure numpy/scipy/pandas — no plotting, no file I/O, no
multiprocessing, no numba. That is what lets the identical code run under
CPython, pytest and Pyodide in the browser, so there is one engine and one set
of tests rather than two implementations that can drift.

**Browser-only, with nothing leaving the machine**, is the deliberate trade.
Real drillhole data makes data egress a hard conversation in mining, and a
static page with a vendored runtime avoids it entirely. The cost is no shared
server state: projects travel as files. The Python runtime is vendored rather
than loaded from a CDN because a CDN is the first thing a locked-down site
network blocks, and a runtime download would quietly undo the property the
design exists for.

## 8. Open items

- **Exported pitch is unverified.** Dip and dip azimuth are checked against
  Leapfrog; pitch is not, and a reversed pitch gives an ellipsoid correctly
  oriented as a plane with its long axis pointing the wrong way inside it —
  which is exactly what angle coding consumes.
  [`docs/LEAPFROG_CHECK.md`](docs/LEAPFROG_CHECK.md) is a half-hour procedure
  that settles it.
- **The comparison scripts hardcode their input filenames** — edit the
  constants at the top of `scripts/compare_methods.py`,
  `validate_contacts.py` and `tune_lsq.py`, or run them with `--synthetic`,
  which needs no data at all. A `--data-dir` flag would remove this.
- **Domaining.** Leiden communities on the grade-similarity graph, so a
  neighbourhood never crosses a lens boundary. `prospect_filter` is currently
  the only control and it is manual.
- **Geodesic neighbourhoods.** The idea that earns the "LVA" name, but it
  consumes an orientation field — worth building only once the field is good
  enough to iterate on.
- **No measured structural data exists** for the test property, so accuracy is
  judged against logged contact surfaces and a synthetic deposit with a known
  answer. Any oriented core or televiewer data would change what can be
  claimed. Conclusions here come from one sediment-hosted Ag-Pb-Zn geometry and
  are not automatically transferable; each deposit needs its own contact check.

## 9. Sharing the app

The exported CSV and the saved settings file are ordinary files. Sharing the
*app* is harder: a browser will not run it from a folder on disk, so each person
needs the setup in [`docs/APP_SETUP.md`](docs/APP_SETUP.md). Putting the built
`app/` folder on an internal web server would remove that — everyone opens a
link, and their data still never leaves their own machine — but that is a
separate piece of work.
