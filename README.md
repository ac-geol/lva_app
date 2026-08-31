# LVA App — Deriving Strike & Dip from Drillhole Data

Deriving local structural orientation (strike/dip, trend/plunge) from drillhole
assay data alone — no oriented core, no televiewer.

Status: **engine extracted and tested; bake-off complete; compositing built
and off by default, not yet swept.**
Full results and reasoning in [`docs/BAKEOFF.md`](docs/BAKEOFF.md).
Running this on another machine: [`docs/SETUP.md`](docs/SETUP.md).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 63 tests
.venv/bin/python scripts/bakeoff.py --all-prospects
.venv/bin/python scripts/validate_contacts.py
```

---

## 1. Current state

| path | what it is |
|---|---|
| `core/` | The engine. Pure numpy/scipy/pandas — no plotting, no file paths, no multiprocessing or numba, so it can run under Pyodide in a browser later without a rewrite. |
| `core/estimators.py` | The four estimators and the single place their pole/lineation conventions are applied. |
| `core/validation.py` | Split-half stability and spatial coherence — how to judge a field with no measured structure. |
| `core/compositing.py` | Downhole compositing to a longer support, off by default. See [`docs/COMPOSITING.md`](docs/COMPOSITING.md). |
| `core/contacts.py` | Reference surfaces fitted from logged geology in `MPA_Interp`. |
| `viz/`, `scripts/` | Stereonets and the runnable comparisons. Deliberately outside `core/`. |
| `tests/` | 63 tests. Geometry conventions, desurvey against hand-worked trigonometry, and estimator recovery of synthetic fields with known answers. |

The orientation loop is gone — replaced by six `bincount` accumulations. The
full property runs in about 0.1 s per estimator.

---

## 2. What the bake-off found

Every estimator ran over one prepared dataset and one shared neighbourhood
graph. Two controls made it interpretable: a **geometry control** (shape-PCA on
weights carrying no grade information — what the drill pattern alone reports),
and **logged contact surfaces** from `MPA_Interp`, fitted one point per hole.

| estimator | vs geometry control | vs logged contacts | split-half stability |
|---|---|---|---|
| geometry_control | — | 21.9° | 15.0° |
| shape_pca | **2.0°** | 21.5° | 15.5° |
| structure_tensor | 74.6° | **73.5°** | 51.2° |
| edge_tensor | 12.6° | 24.7° | 22.1° |
| **lsq_gradient** | 25.8° | **29.2°** (27.2° tuned) | 36.5° |

60° is the expectation for random axes.

**Shape-PCA is the drill pattern.** Its answer differs from the pure geometry
control by 2.0° across the property and 0.28° at Tom West. Confirmed as
predicted — though with a twist: it scores 21.5° against logged geology, better
than anything else here. Holes are drilled across the interpreted lens, so it
recovers *the geologist's prior*. Circular, not useless, and useless for the
stated goal.

**The graph structure tensor failed.** It scored 73.5° against logged
contacts — worse than random — while reporting high confidence. Its poles lie
systematically within the plane of the true contacts. On a **synthetic,
noise-free lens** placed on the real drill layout it recovered the pole to only
36–57°, so this is not a nugget problem: `T = Σ w δ² û ûᵀ` averages over
whatever edge directions the drill programme provided, and that average
estimates a gradient only when those directions are isotropic. Measured
sampling anisotropy here is λ_min/λ_max ≈ 0.09–0.24.

The previous recommendation in this README — that the structure tensor fixes
the geometry bias "directly" — was wrong. It does not escape the bias.

**What works is to solve rather than average.** Least-squares gradient
reconstruction minimises `Σ w (δ_ij − ∇g·û_ij)²`, whose normal equations
contain the sampling-direction tensor explicitly and therefore invert it away.
On the synthetic lens: 57.3° → **9.1°**.

---

## 3. Two ways to fool yourself, both confirmed here

**Pole concentration is the wrong selection metric.** Woodcock S1/C — proposed
in the earlier version of this README — ranks shape-PCA first, because the drill
programme is globally consistent and so measuring it produces a tight
stereonet. Tightness rewards the artifact.

**Smoothing manufactures structure.** Across the tuning sweep, added score
smoothing improved every internal metric while monotonically degrading the
external one. At five passes the field reports confidence 0.92 and spatial
coherence 1.3° while sitting 72.6° from logged geology — worse than random.
Internal confidence and external truth move in opposite directions.

Selection must use split-half stability and the contact check. Both are cheap
and both are implemented.

---

## 4. Next steps

1. **Reduce nugget at source.** The synthetic benchmark isolates noise as the
   binding constraint (9° clean → 32° at noise 0.5), and the real data sits at
   27°. Downhole compositing to a longer support *before* differencing is the
   highest-value change. Post-hoc smoothing is not — it demonstrably destroys
   the signal.

   **Compositing is now built and tested** (`core/compositing.py`, 25 tests),
   and off by default. It has **not been swept yet**: `min_neighbors` is an
   absolute sample count, so raising the composite length silently guts the
   node set — at 8 m the field collapses to 8 nodes in 3 holes spanning 22 m.
   The fix is to threshold on metres of neighbouring core rather than on sample
   count, which is invariant under compositing. Full reasoning and the numbers
   are in [`docs/COMPOSITING.md`](docs/COMPOSITING.md) §4. **Do that before
   drawing any conclusion from a composited run.**
2. **Domaining.** Leiden communities on the grade-similarity graph, so a
   neighbourhood never crosses a lens boundary. Currently `prospect_filter` is
   the only control and it is manual.
3. **Geodesic neighbourhoods.** Still the idea that earns the "LVA" name, but
   it consumes an orientation field — worth building only once the field is
   good enough to be worth iterating on.
4. **Then the app.** Deferred deliberately: the science is not settled enough
   to wrap a UI around. See §5.

---

## 5. Platform decision

Python-first, browser later. `core/` is written to keep that door open at
near-zero cost: pure numpy/scipy/pandas, no plotting or file I/O inside, no
multiprocessing and no numba. If it runs under plain CPython it runs under
Pyodide, so "adopt the browser" becomes a UI project rather than a rewrite.

The target is a team of technical professionals who do not write Python, which
a CLI does not serve. That is an accepted deferral, not a solved problem — and
it means "working well" has to mean *the science is settled and the API is
stable*, not *the scripts run*.

Constraints recorded when that decision was made:

- Browser-only (static, Pyodide in a Web Worker) means **no client data ever
  leaves the machine**, which matters more than compute for real client
  drillhole data. Cost: no shared server state — projects travel as files.
- A server backend buys shared results and full-speed numpy, and costs a data
  egress conversation that is often a hard no in mining.
- The geodesic refinement is the one step likely too slow for WASM; plan it as
  an opt-in mode, not a slider.

---

## 6. Open questions

- **No measured structural data** (confirmed: none available). The logged
  contact surfaces in `MPA_Interp` are the best available substitute and are
  now wired in, but 12 of 26 candidate surfaces were rejected for sitting on a
  single drill fence. Any oriented core or televiewer data would change what
  can be claimed here.
- ~~**Is 27° useful?**~~ **Settled.** The objective is discerning major trends
  and using them for **LVA angle coding in estimation** — not picking veins —
  and 27° is accepted as adequate for that. Accuracy is therefore no longer the
  blocking constraint. Note the tension this creates: angle coding wants a
  smooth, continuous field, but §3 shows apparent smoothness and external truth
  move in *opposite* directions here. Any "smooth it for coding" step needs the
  contact check, not eyeballing.
- **Which deposit is the real target** — MacPass is the test case, and it is a
  sediment-hosted Ag-Pb-Zn geometry. Conclusions above are not automatically
  transferable to a deposit with a different geometry or a different grade
  distribution; each one needs its own contact check.

## Files

This repo carries **no drillhole data**. `.gitignore` is deny-by-default for
data: every `*.csv` and `data/` is ignored, and only the aggregate-only tables
below are re-admitted by explicit exception.

- `out/` — bake-off figures and the aggregate summary tables
  (`bakeoff_summary.csv`, `lsq_tuning.csv`, `structure_tensor_sweep.csv`,
  `contact_validation.csv`): angles, counts and timings only, no coordinates
  and no hole IDs. These are the evidence behind `docs/BAKEOFF.md`.
- Everything else the scripts write — `orientations_*.csv`, `contacts.csv`,
  `contact_planes.csv` — carries coordinates or hole IDs and is ignored.

**Bring your own data.** The scripts currently expect the four MacPass tables
(`MPA_Samples_BD`, `MPA_Collar`, `MPA_Survey`, `MPA_Interp`) in the repo root;
MacPass is public data, and the filenames are still hardcoded in
`scripts/bakeoff.py`, `validate_contacts.py` and `tune_lsq.py`. `core/schema.py`
resolves common column aliases and `describe_mapping()` previews the resolution
for any CSV.

The MacPass numbers quoted throughout this README and `docs/BAKEOFF.md` stand as
a written record, but **they are not reproducible from a clean clone** — the
tables they were computed from are not in the repo.

The 63 tests are self-contained and need no data: `git clone` + `pytest` passes.
