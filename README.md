# LVA App

Deriving local structural orientation — strike and dip — from drillhole assay
data alone. No oriented core, no televiewer.

The intended use is **LVA angle coding for grade estimation**: giving a search
ellipsoid a local orientation that follows the mineralisation rather than a
single global direction.

Status: **engine and browser app working and tested.**

**[Setup and usage guide →](docs/APP_SETUP.md)** — start here. It assumes no
Python, and covers everything from installing the tools to reading the result.

---

## What is here

| path | what it is |
|---|---|
| `app/` | A browser interface — runs the engine on your own machine, with no server and no data leaving the browser. |
| `core/` | The engine. Pure numpy/scipy/pandas — no plotting, no file paths. |
| `scripts/` | The app build, and runnable method comparisons. |
| `viz/` | Stereonets. Deliberately outside `core/`. |
| `tests/` | 62 tests. Self-contained — they need no data. |

## Quick start

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 62 tests, no data required
```

Then follow [`docs/APP_SETUP.md`](docs/APP_SETUP.md) to download the browser
runtime and start the app.

## Methods

Two ship, and they answer different questions:

- **`shape_pca`** — principal component analysis on the positions of the
  high-grade samples around each point. The default.
- **`lsq_gradient`** — least-squares reconstruction of the local grade
  gradient, which removes the drill geometry by construction.

Every run also computes what the drill pattern alone would report, with grades
stripped out, and reports the angle between the two. That number is the check on
whether a result is measuring rock or measuring drilling.

## Data

**This repo carries no drillhole data**, and none is ever committed —
`.gitignore` ignores every `*.csv` and `data/` outright. Bring your own.

The app takes **one CSV: one row per sample interval, already desurveyed**, with
hole ID, from/to, X/Y/Z and one assay column. Column names do not have to match
anything; the app shows you what it matched and lets you correct it.

## Running the comparison scripts

For anyone working on the engine rather than using the app. These take the raw
three-table export — samples, collars and surveys — plus logged intervals for
the contact check, and their paths are constants at the top of each script:

```bash
.venv/bin/python scripts/compare_methods.py --synthetic   # no data needed
.venv/bin/python scripts/compare_methods.py               # your own tables
.venv/bin/python scripts/validate_contacts.py             # the external check
.venv/bin/python scripts/tune_lsq.py                      # parameter sweeps
```

`--synthetic` builds a folded deposit whose true orientation is known at every
sample, so a fresh clone can produce a full comparison with nothing to supply.

## Reasoning and open items

[`CLAUDE.md`](CLAUDE.md) — what was decided and why, what is parked, and what is
still open. [`docs/LEAPFROG_CHECK.md`](docs/LEAPFROG_CHECK.md) is the one
verification still outstanding: **exported pitch is not yet confirmed** against
Leapfrog, though dip and dip azimuth are.
