# LVA App

Deriving local structural orientation — strike and dip — from drillhole assay
data alone. No oriented core, no televiewer.

The intended use is **LVA angle coding for grade estimation**: giving a search
ellipsoid a local orientation that follows the mineralisation rather than a
single global direction.

Status: **engine working and tested. A browser interface is built and under
test** on the `feature/browser-app` branch.

---

## What is here

| path | what it is |
|---|---|
| `core/` | The engine. Pure numpy/scipy/pandas — no plotting, no file paths. |
| `app/` | A browser interface — runs the engine locally in the browser, with no server and no data leaving the machine. **Not on `main` yet**; see `feature/browser-app`. |
| `scripts/` | Runnable comparisons and the app build. |
| `viz/` | Stereonets. Deliberately outside `core/`. |
| `tests/` | 71 tests. Self-contained — they need no data. |
| `docs/` | Setup, results and reasoning. |

## Quick start

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 71 tests, no data required
```

The browser app is not on `main` yet. It lives on the `feature/browser-app`
branch, along with its own setup guide (`docs/APP_SETUP.md`):

```bash
git switch feature/browser-app
```

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

**This repo carries no drillhole data.** `.gitignore` is deny-by-default: every
`*.csv` and `data/` is ignored, with explicit exceptions only for aggregate
tables that hold angles, counts and timings — no coordinates, no hole IDs.

Bring your own. The app takes one desurveyed CSV; the scripts take the three
raw tables.

## Documentation

| | |
|---|---|
| [`docs/SETUP.md`](docs/SETUP.md) | Running the engine on another machine. Start here. |
| [`docs/METHOD_COMPARISON.md`](docs/METHOD_COMPARISON.md) | Which method ships and why. |
| [`docs/COMPOSITING.md`](docs/COMPOSITING.md) | Downhole compositing, and why it is out of scope. |
| [`docs/BACKGROUND.md`](docs/BACKGROUND.md) | The fuller reasoning behind the current design. |
| [`docs/TODO.md`](docs/TODO.md) | Open items and parked decisions. |

## Known open items

- **Exported pitch is not yet verified** against Leapfrog. Dip and dip azimuth
  are. The procedure is on the `feature/browser-app` branch.
- **Which method is right is not fully settled.** `shape_pca` is more accurate
  on the evidence available but closely resembles the drill pattern; the
  experiment that would decide it is in [`docs/TODO.md`](docs/TODO.md) §3.
- **No measured structural data** exists for the test property, so accuracy is
  judged against logged contact surfaces and a synthetic deposit with a known
  answer.
