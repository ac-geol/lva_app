# Downhole compositing

Status: **built, tested, off by default — and not the recommended path.**

**Composite upstream.** Geologists already composite in the software that holds
their data (Leapfrog, Datamine, Vulcan), and this project does not intend to
compete with those tools. `core/compositing.py` stays as a reference
implementation and as the place the length-conservation property is pinned by
tests; the planned composite-length sweep has been dropped.

`composite_length_m: None` reproduces the pre-compositing pipeline exactly —
verified byte-for-byte against `out/method_comparison_summary.csv`, every
scientific column identical.

**§4 matters more now, not less.** Compositing upstream means this pipeline
receives data of arbitrary, unknown support, and `min_neighbors` is an absolute
sample count calibrated at MacPass's ~1.27 m assays. Feeding it composites made
elsewhere hits exactly the collapse described there — with nothing in the output
revealing that the filter, not the data, caused it. Read §4 before running this
pipeline on any composited table, wherever the compositing was done.

---

## 1. Why

Both methods here work from finite differences of the score between holes,

```
delta_ij = (s_j - s_i) / d_ij
```

so each node's noise enters the gradient fit as `sigma_eps / d`. There are only
two levers. The baseline `d` is already as long as the geology allows
(`radius_m: 120`, `distance_sigma_m: 80`) and pushing it further starts
averaging over genuinely different structure. That leaves `sigma_eps`.

And `sigma_eps` is large because the support is tiny. MacPass assay intervals:

```
mean 1.27 m   median 1.34 m   modal 1.5 m (9430) and 1.0 m (5930)
32,546 intervals across 432 holes
```

We are differencing **1.3 m supports across ~100 m baselines**. At that support
in an Ag-Pb-Zn system an individual assay is mostly nugget.

The synthetic benchmark in `docs/METHOD_COMPARISON.md` is what makes this the priority
rather than a guess: `lsq_gradient` scores **9.1° on a clean lens and 32° at
noise 0.5**, and the real data sits at 27.2°. The method is already near its
noise-limited ceiling, and no amount of method work moves a number set by
`sigma_eps`.

## 2. Why this is not the smoothing that failed

Superficially both are "average the data first", and `docs/METHOD_COMPARISON.md` §4 shows
smoothing degrading the contact error from 27.2° to 72.6° — worse than random —
while every internal metric improved. The distinction is mechanical, not a
matter of degree:

| | what it averages | effect on a difference `s_j - s_i` |
|---|---|---|
| `smoothing.smooth_scores` | scores across the **3-D graph whose edges are then differenced** | each node's value bleeds into its neighbours, so the difference becomes partly a difference of **shared terms** — it manufactures agreement between exactly the pair being measured |
| `compositing.composite_downhole` | raw grades **along the hole only, before the graph exists** | with `exclude_same_hole_neighbors: True`, i and j are always in different holes, so **no term is ever shared** across a difference |

Compositing changes the support. It does not correlate the neighbours.

The genuine cost is different and must be respected: structure thinner than the
composite length is averaged away. The useful range is bounded above by the
thickness of the thing being resolved, so the sweep should find a **minimum**,
not a monotone improvement. A monotone result is itself a finding — it would
mean nugget was not the binding constraint and the synthetic benchmark misled
us.

## 3. What was built

`core/compositing.py` — `composite_downhole(samples, length_m, ...)`. Fully
vectorized, pure numpy/pandas, so it holds the `core/` Pyodide constraint.

- **Fixed grid, anchored per run.** A run is a contiguous stretch of assayed
  ground; a source interval straddling a boundary contributes its overlapping
  length to both sides.
- **Runs break on wide gaps** (default tolerance: one composite length).
  MacPass has 959 real downhole gaps, the largest 335 m. Compositing across one
  would invent a value for ground nobody assayed. Narrow gaps stay inside a
  composite and lower its `coverage`.
- **`From_m`/`To_m` bound the assayed extent; `interval_m` is the assayed
  length.** These differ when a composite spans an internal gap, and
  `interval_m` is the one that matters — `neighbors.edge_weights` weights a
  neighbour by it.
- **Short final cells merge** into the preceding composite. A run shorter than
  `length_m` is kept whole rather than discarded.
- **Raw grades, never the score.** The score is `mean(z(log1p(grade)))` and log
  of a mean is not the mean of a log, so compositing must precede scoring.
  `prepare()` runs it before desurvey, which is also where it belongs
  physically — it is a `From_m`/`To_m` operation.

### Verified

- **Metal conservation is exact** — max relative error 2.6e-16 across
  L ∈ {2, 3, 5, 8, 12, 20}.
- **Null assays are a convention, not a bug.** A whole-table check initially
  showed ~1e-3 error; it traced entirely to null assays and ranked exactly by
  null count (Ag 1113 > Pb 87 > Zn 82). A composite's value stands for its whole
  assayed length, so unassayed ground inherits the grade around it rather than
  counting as zero. Pinned by test.
- 25 tests in `tests/test_compositing.py`, including a variance check that
  confirms sd falls by ~sqrt(k) on a flat noise field.

---

## 4. The open issue — `min_neighbors` blocks the sweep

**This is the recommendation for the next round of work.**

`min_neighbors` is an **absolute sample count**, checked at
`core/estimators.py:81`:

```python
ok = edges.neighbor_counts() >= cfg['min_neighbors']    # 100
```

Compositing reduces the number of samples ~k-fold while sampling *exactly the
same rock*. Neighbour counts fall with it, and nodes begin failing a filter
that has nothing to do with data quality. Which filter rejects what:

```
    L  active  fail nbr  fail hole   pass  %kept  holes  extent m
 None    4882       856        212   4026  82.5%    214     20232
    2    3157       732        165   2425  76.8%    197     20216
    3    2120       803        116   1317  62.1%    158     20194
    5    1288       931         68    357  27.7%     86     20035
    8     821       813         45      8   1.0%      3         22
   12     559       559         45      0   0.0%      0         0
```

`min_holes: 3` behaves correctly throughout — it is a count of *distinct
contributing holes*, which is support-invariant, and it rejects fewer nodes as L
rises. `min_neighbors` is what collapses the field.

**It is not merely "fewer nodes."** At L=none the surviving field spans 214
holes across 20 km of property. At L=8 it is **8 nodes, in 3 holes, spanning 22
metres**. A contact error computed there is not a noisier estimate of the same
quantity — it is a different quantity, measured on a tiny patch of the densest
drilling. Sweeping L naively would report that compositing catastrophically
fails above 3 m, and nothing in the output table would reveal that the filter,
not the method, caused it.

### Recommended fix: threshold on metres of core, not on sample count

The filter is counting the wrong unit. What it is trying to guarantee is enough
rock behind each estimate, and a sample count only proxies for that while
support is fixed — which is precisely the assumption compositing breaks.
Compositing conserves assayed length exactly, so a threshold in metres is
nearly invariant by construction:

```
    L  med nbr count  med nbr core m
 None            290             297
    2            180             352
    3            124             360
    5             74             361
    8             51             384
   12             34             391
```

Count collapses 8.5×. Metres of core drifts 1.3×.

So: add a `min_neighbor_length_m` gate (≈127 m — what `min_neighbors: 100`
meant at 1.27 m support) and use it in place of the count when sweeping.
`min_holes: 3` stays as-is.

### The residual drift is a second, unfixable confound

Metres-of-core still creeps up ~30%. That is `is_active` — the top 15% by score.
Compositing changes *which* material that is: isolated high-grade spikes average
down, sustained mineralized intervals survive, and the active set shifts toward
better-drilled ground. For trend detection that shift is arguably an
improvement, but it is a change in what is being estimated, not just how
precisely. It cannot be filtered away, so any sweep must report node counts and
spatial extent beside every error figure.

## 5. When the sweep runs

- L ∈ {none, 2, 3, 5, 8, 12}, scored on **external metrics only**: contact error
  from `scripts/validate_contacts.py`, plus split-half stability. Internal
  metrics reward the artifact — that is the standing lesson of §4 of
  `docs/METHOD_COMPARISON.md`.
- Run the **synthetic lens** too. Truth is known, geometry is fixed, noise is a
  dial, and there is no active-set confound in the way. If compositing works
  mechanically, the noise-0.5 case should move from 32° toward the clean 9.1°.
- `gradient_r2` already rises monotonically with L (0.25 → 0.32), which is
  consistent with the mechanism working. It is an internal metric and proves
  nothing on its own.
