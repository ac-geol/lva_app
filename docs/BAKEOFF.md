# Estimator bake-off — MacPass, August 2026

Ran all candidate estimators over the MacPass Ag-Pb-Zn dataset (32,546 samples,
559 collars, 316 holes with usable geometry, 4,882 active nodes at the 0.85
score quantile). Every estimator consumes the **same** prepared dataset and the
**same** neighbourhood graph, so the comparison isolates the estimator.

Reproduce with:

```bash
python scripts/bakeoff.py --all-prospects   # stereonets + internal diagnostics
python scripts/validate_contacts.py         # external check vs logged geology
python scripts/tune_lsq.py                  # parameter sweep for the winner
```

---

## 1. The two controls that made this interpretable

**Geometry control.** Shape-PCA run on weights that carry *no grade
information* — distance and interval length only. It is what the drill pattern
alone reports. Any estimator that lands near it is measuring drilling.

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

| estimator | vs geometry control | vs logged contacts | split-half stability | spatial coherence (null) |
|---|---|---|---|---|
| geometry_control | — | 21.9° | 15.0° | 17.0° (47.4°) |
| shape_pca | **2.0°** | 21.5° | 15.5° | 15.7° (46.6°) |
| structure_tensor | 74.6° | **73.5°** | 51.2° | 54.4° (58.8°) |
| edge_tensor | 12.6° | 24.7° | 22.1° | 21.8° (48.2°) |
| lsq_gradient | 25.8° | 29.2° | 36.5° | 42.2° (50.7°) |

All angles in degrees; **60° is the expectation for random axes.**

### Shape-PCA is the drill pattern, confirmed

Shape-PCA's answer differs from the pure geometry control by **2.0° across the
property, and 0.28° at Tom West.** Grade weighting moves the result by a
fraction of a degree. This settles README §2 empirically: the estimator carries
essentially no information that is not already in the hole locations.

The subtlety is that this is *not the same as being wrong*. Shape-PCA scores
21.5° against the logged contacts — far better than random. But so does the
control, at 21.9°. Holes are drilled across the interpreted lens, so a PCA of
the drill pattern recovers **the geologist's prior**, not new information. It
is circular, not useless — and useless for the stated goal of deriving
orientation from assay data alone.

### The structure tensor failed, and not from noise

The README's recommended estimator scored **73.5° against the logged
contacts — worse than random.** Being consistently ~75° off is not noise;
noise gives 60°. Its poles lie systematically *within* the plane of the true
contacts, while `planarity` reported a median of 0.82 (of a 0.67 ceiling).
Confident, and orthogonal to reality.

Two hypotheses were tested:

1. *Nodes and neighbours are both drawn from the top 15% of grades, so every
   gradient is measured inside ore and never across the ore/waste contact.*
   Real, but not the cause — widening the neighbour pool to all samples moved
   it only 71.5° → 65.0°.

2. *The estimator is broken for this sampling geometry.* Confirmed. Placing a
   **synthetic, noise-free lens** on the real MacPass drill layout — a case
   with a known exact answer and no nugget at all — the structure tensor
   recovered the pole to only **36–57°**, while reporting `planarity_norm =
   1.00`.

The cause is structural. `T = Σ w δ² û ûᵀ` averages over whatever edge
directions the drill programme happened to provide. That average estimates the
gradient direction only when the directions û are **isotropic**, and drillhole
sampling is the opposite of isotropic — the measured sampling-direction tensor
has λ_min/λ_max ≈ 0.09–0.24. The direction of largest observed contrast is
therefore partly a statement about where the other holes are.

**README §3a's central claim — that the structure tensor fixes the geometry
bias "directly" — is wrong.** It does not escape the bias; it is exposed to it
differently, and worse.

---

## 3. What actually works: least-squares gradient reconstruction

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

| lens thickness | noise | structure_tensor | lsq_gradient |
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
contacts, 33.1° split-half stability, 24.4° from the geometry control.

`gradient_r2` sits at 0.20–0.25: a single linear gradient explains about a
fifth of the observed grade contrast. The rest is nugget. That is a low number
and it is an honest one.

---

## 4. Two metrics that would have picked the wrong estimator

**Pole concentration (Woodcock S1/C), which README §5 proposed as the selection
metric, is actively misleading here.** The drill programme is globally
consistent, so an estimator that measures drilling produces a *tight* stereonet.
Shape-PCA scores S1 = 0.61 and the structure tensor 0.40 — the metric ranks the
known-circular estimator first. Tightness rewards the artifact.

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
stereonet tightness, spatial coherence, or the estimator's own confidence
score will confidently ship garbage.

Selection must use split-half stability and the contact check. Both are cheap.

---

## 5. Where this leaves the project

- `lsq_gradient` is the only estimator that both derives its answer from grade
  rather than geometry (25.8° from the control) and beats random against
  logged geology (27.2°). It is the one to build on.
- 27° is honest but not yet good. The synthetic benchmark says nugget is the
  binding constraint (9° clean → 32° at noise 0.5), so the highest-value next
  step is **reducing noise at source — downhole compositing to a longer
  support before differencing — not smoothing afterwards**, which demonstrably
  destroys the signal.
- Every estimator remains available and tested; `shape_pca` is worth keeping
  precisely because it reproduces the drilling prior, which makes it a useful
  null.
