# LVA method comparison — MacPass, August 2026

Two methods ship, and they answer different questions:

| method | what its tensor is built from | ships as |
|---|---|---|
| `lsq_gradient` | a **solved** local grade gradient | the default |
| `shape_pca` | a covariance of neighbour **positions** | the drilling null |

Alongside them the comparison plots a **drill-pattern reference** — shape-PCA on
weights carrying no grade information at all. It is not a method. It is what the
drill pattern reports with the geology removed, and it is what makes the two
methods interpretable: a method that lands on top of it is measuring drilling.

Run over the MacPass Ag-Pb-Zn dataset (32,546 samples, 559 collars, 316 holes
with usable geometry, 4,882 active nodes at the 0.85 score quantile). Both
methods consume the **same** prepared dataset and the **same** neighbourhood
graph, so the comparison isolates the method.

Reproduce with:

```bash
python scripts/compare_methods.py --all-prospects   # stereonets + internal diagnostics
python scripts/validate_contacts.py                 # external check vs logged geology
python scripts/tune_lsq.py                          # parameter sweep for the default
```

> These numbers were produced against MacPass tables that are **not** in this
> repository (see `docs/TODO.md`). They stand as a written record and as
> reasoning; they are not reproducible from a clean clone.

---

## 1. The two references that made this interpretable

**Drill-pattern reference.** Shape-PCA run on weights that carry *no grade
information* — distance and interval length only. It is what the drill pattern
alone reports. Any method that lands near it is measuring drilling. Reported as
`vs_reference_deg`.

**Logged contact surfaces.** `MPA_Interp` picks named stratigraphic units
(Tom Massive Sulphides in 75 holes, Jason South Upper Lens in 29, …). Taking
each unit's top and base **once per hole** and fitting a plane gives a
reference surface that is independent of the assay data and immune to downhole
collinearity. 12 of 26 candidate surfaces were rejected because their
contributing collars sit on a single drill fence (`collar_aspect < 0.15`),
which cannot constrain a plane.

Neither is ground truth. Together they bracket the question.

---

## 2. Headline result

Five candidates were evaluated. Two ship; two were rejected and their code has
since been removed; the fifth is the reference.

| candidate | status | vs drill-pattern reference | vs logged contacts | split-half stability | spatial coherence (null) |
|---|---|---|---|---|---|
| drill-pattern reference | reference | — | 21.9° | 15.0° | 17.0° (47.4°) |
| `shape_pca` | **ships** (as the null) | **2.0°** | 21.5° | 15.5° | 15.7° (46.6°) |
| `lsq_gradient` | **ships** (default) | 25.8° | **29.2°** (27.2° tuned) | 36.5° | 42.2° (50.7°) |
| `structure_tensor` | removed | 74.6° | 73.5° | 51.2° | 54.4° (58.8°) |
| `edge_tensor` | removed | 12.6° | 24.7° | 22.1° | 21.8° (48.2°) |

All angles in degrees; **60° is the expectation for random axes.**

### Shape-PCA is the drill pattern, confirmed

Shape-PCA's answer differs from the pure drill-pattern reference by **2.0°
across the property, and 0.28° at Tom West.** Grade weighting moves the result
by a fraction of a degree. The method carries essentially no information that
is not already in the hole locations.

The subtlety is that this is *not the same as being wrong*. Shape-PCA scores
21.5° against the logged contacts — far better than random. But so does the
reference, at 21.9°. Holes are drilled across the interpreted lens, so a PCA of
the drill pattern recovers **the geologist's prior**, not new information. It
is circular, not useless — and useless for the stated goal of deriving
orientation from assay data alone.

It ships anyway, and deliberately: a method that faithfully reproduces the
drilling prior is the most useful null available. If `lsq_gradient` ever
collapses onto `shape_pca`, that is the signal it has stopped measuring rock.

### The structure tensor failed, and not from noise

It scored **73.5° against the logged contacts — worse than random.** Being
consistently ~75° off is not noise; noise gives 60°. Its poles lay
systematically *within* the plane of the true contacts, while `planarity`
reported a median of 0.82 (of a 0.67 ceiling). Confident, and orthogonal to
reality.

Two hypotheses were tested:

1. *Nodes and neighbours are both drawn from the top 15% of grades, so every
   gradient is measured inside ore and never across the ore/waste contact.*
   Real, but not the cause — widening the neighbour pool to all samples moved
   it only 71.5° → 65.0°.

2. *The method is broken for this sampling geometry.* Confirmed. Placing a
   **synthetic, noise-free lens** on the real MacPass drill layout — a case
   with a known exact answer and no nugget at all — it recovered the pole to
   only **36–57°**, while reporting `planarity_norm = 1.00`.

The cause is structural. `T = Σ w δ² û ûᵀ` averages over whatever edge
directions the drill programme happened to provide. That average estimates the
gradient direction only when the directions û are **isotropic**, and drillhole
sampling is the opposite of isotropic — the measured sampling-direction tensor
has λ_min/λ_max ≈ 0.09–0.24. The direction of largest observed contrast is
therefore partly a statement about where the other holes are.

### Why both rejects were removed rather than kept

Neither had a use that `shape_pca` did not already serve better. `edge_tensor`
at 12.6° from the reference is a second, noisier way of measuring the drill
pattern; `structure_tensor` is actively wrong on this sampling geometry. Two
methods and one reference is the whole useful set, and offering five choices
where three are traps is worse than offering two.

**The evidence survives the deletion.** The claim that solving beats averaging
under anisotropic sampling is pinned by
`tests/test_conventions.py::test_lsq_gradient_beats_naive_tensor_averaging_on_anisotropic_sampling`,
which builds the averaged tensor `Σ w δ² û ûᵀ` inline — the method being
beaten, rather than a shipped estimator — and asserts the solved gradient is at
least twice as accurate. Deleting the production code did not delete the
finding. `scripts/sweep.py`, which existed only to ask whether the structure
tensor could be rescued, was removed with it.

---

## 3. What ships as the default: least-squares gradient reconstruction

Stop averaging. Solve. Minimising

```
    Σ_j w_ij (δ_ij − ∇g · û_ij)²
```

gives the normal equations

```
    (Σ_j w_ij û_ij û_ijᵀ) ∇g = Σ_j w_ij δ_ij û_ij
```

where the matrix on the left **is** the sampling-direction tensor. Inverting it
deconvolves the drill geometry instead of averaging over it.

On the synthetic-lens benchmark, at identical settings:

| lens thickness | noise | averaged tensor | `lsq_gradient` |
|---|---|---|---|
| 10 m | 0.0 | 57.3° | **9.1°** |
| 25 m | 0.0 | 42.3° | **12.6°** |
| 60 m | 0.0 | 35.9° | **16.4°** |
| 10 m | 0.5 | 58.5° | **32.1°** |

Confidence comes from `gradient_r2` — the share of observed grade contrast one
linear gradient explains — not from an eigenvalue ratio of a rank-one tensor,
which would always be 1.0. `sampling_conditioning` reports when the drill
pattern cannot constrain a gradient at all.

### Tuned result on real data

Best configuration: **radius 180 m, no score smoothing** → 27.2° from logged
contacts, 33.1° split-half stability, 24.4° from the drill-pattern reference.

`gradient_r2` sits at 0.20–0.25: a single linear gradient explains about a
fifth of the observed grade contrast. The rest is nugget. That is a low number
and it is an honest one.

---

## 4. Two metrics that would have picked the wrong method

**Pole concentration (Woodcock S1/C) is actively misleading here.** The drill
programme is globally consistent, so a method that measures drilling produces a
*tight* stereonet. Shape-PCA scores S1 = 0.61 and the structure tensor 0.40 —
the metric ranks the known-circular method first. Tightness rewards the
artifact.

**Smoothing is worse.** Across the tuning sweep, added score smoothing improved
every internal appearance metric while degrading the external one, monotonically:

| smoothing | gradient_r2 | spatial coherence | contact error |
|---|---|---|---|
| none | 0.20 | 42° | **27.2°** |
| 2 × α=0.5 | 0.26 | — | 30.6° |
| 3 × α=0.7 | 0.62 | 2.7° | 56.2° |
| 5 × α=0.7 | 0.92 | 1.3° | 72.6° |

At five passes the field looks superb — coherence 1.3°, confidence 0.92 — and
is worse than random against the logged contacts. Internal confidence and
external truth move in **opposite directions** under smoothing. Tuning on
stereonet tightness, spatial coherence, or the method's own confidence score
will confidently ship garbage.

Selection must use split-half stability and the contact check. Both are cheap.

---

## 5. Where this leaves the project

- `lsq_gradient` is the only method that both derives its answer from grade
  rather than geometry (25.8° from the reference) and beats random against
  logged geology (27.2°). It is the default and the one to build on.
- `shape_pca` ships as the null, not as a competitor. Its value is diagnostic.
- 27° is honest but not yet good, and it is **accepted as adequate** for the
  stated goal of major-trend LVA angle coding. The synthetic benchmark says
  nugget is the binding constraint (9° clean → 32° at noise 0.5), so the
  highest-value lever is reducing noise at source — **compositing to a longer
  support before differencing, done upstream in whatever software already holds
  the data**, not smoothing afterwards, which demonstrably destroys the signal.
- Compositing upstream means this pipeline receives data of **arbitrary,
  unknown support**, which makes `min_neighbors` — an absolute sample count
  calibrated at MacPass's ~1.27 m assays — the next thing to fix. See
  [`COMPOSITING.md`](COMPOSITING.md) §4.
