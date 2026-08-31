# Setup — running this on another machine

The repo carries **no drillhole data** — code, tests, docs and aggregate result
tables only. The 63 tests are self-contained, so `git clone` + `pytest` proves
the engine works with nothing else to fetch. Budget about five minutes.

To reproduce the bake-off numbers you supply your own tables — see §5.1. The
MacPass results quoted in `docs/BAKEOFF.md` stand as a written record but are
**not reproducible from a clean clone**.

**Windows, macOS and Linux all work.** `core/` uses no OS-specific code — no
`os` calls, no `subprocess`, no multiprocessing, no POSIX-only modules — every
path is relative and built with `pathlib`, plots are rendered headless via
matplotlib's `Agg` backend, and all console output is plain ASCII, so
redirecting a script to a file will not hit a Windows encoding error. All five
pinned dependencies ship Windows wheels for CPython 3.11 (x64 and ARM64).

Commands below are given for **PowerShell** and for **bash** (macOS/Linux).
The one difference that matters everywhere: the interpreter is
`.venv\Scripts\python.exe` on Windows and `.venv/bin/python` elsewhere.

---

## 1. Prerequisites

| need | check | if missing |
|---|---|---|
| Python 3.11 | `python3 --version` | see §1.1 |
| git | `git --version` | preinstalled on macOS/Linux; else git-scm.com |
| GitHub access | `gh auth status` | see §2 — **the repo is private** |

Python 3.12/3.13 will very likely work, but 3.11.15 is what everything was
verified against. If you hit anything strange, match the version first.

### 1.1 Installing uv (recommended)

`uv` handles both Python and the virtualenv, so you do not need a system
Python 3.11 at all:

PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

bash (macOS/Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then **open a new terminal** so the updated PATH is picked up. (On bash you can
instead `export PATH="$HOME/.local/bin:$PATH"`.)

Plain `python -m venv` + `pip` works equally well — see §3.2. On Windows,
`winget install --id Python.Python.3.11` is the simplest way to get a system
Python if you would rather not use uv.

---

## 2. Clone

**The repo is private**, so an unauthenticated `git clone` will fail with a
confusing 404. Authenticate first:

```bash
gh auth login          # GitHub CLI -- easiest; choose HTTPS
gh repo clone ac-geol/lva_app
cd lva_app
```

Without `gh`, use a personal access token with `repo` scope, or an SSH key
registered on the account:

```bash
git clone git@github.com:ac-geol/lva_app.git
cd lva_app
```

### Which branch?

```bash
git branch -a
```

| branch | what it is |
|---|---|
| `main` | The finished bake-off. The engine, 38 tests, and every result written up in `docs/BAKEOFF.md`. |
| `feature/downhole-compositing` | `main` plus downhole compositing: `core/compositing.py`, 25 more tests, and `docs/COMPOSITING.md`. Compositing is **off by default** (`composite_length_m: None`), so this branch reproduces `main`'s numbers exactly unless you turn it on. |

`main` is checked out by default. To get the compositing work:

```bash
git checkout feature/downhole-compositing
```

---

## 3. Environment

### 3.1 With uv (recommended)

PowerShell:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

bash:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

uv downloads Python 3.11 itself if the machine does not have it, on every
platform.

### 3.2 Without uv

PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

bash:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

**From here on, commands are written with `.venv/bin/python`. On Windows,
substitute `.venv\Scripts\python.exe`** — that is the only change needed.

There is no `pip install -e .` step and no packaging — `core/` is imported by
path. That is why every command below runs `.venv/bin/python` from the repo
root rather than activating the venv; either works, but running from the root
is what makes `import core` resolve.

---

## 4. Verify

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q     # Windows
```

```bash
.venv/bin/python -m pytest tests/ -q              # macOS / Linux
```

Expect **38 passed** on `main`, **63 passed** on `feature/downhole-compositing`.

**The first run takes ~18 seconds; every run after that is ~2.** That is Python
compiling bytecode on the cold clone, not a hang. Verified from a clean clone
of this repo.

If this passes, the engine is good — every geometry
convention, the desurvey trigonometry, and the estimators' recovery of
synthetic fields with known answers are all covered.

If it fails, stop here; nothing downstream will mean anything.

---

## 5. Run things

All commands run from the repo root. Runtimes below were measured on the
development machine (Apple silicon, Python 3.11.15).

### 5.1 Bring your own data

The repo ships no drillhole tables, so every script in this section needs data
you supply. You need three tables, plus a fourth for the contact check:

| table | needs columns |
|---|---|
| samples / assays | hole ID, from, to, one or more grade columns |
| collars | hole ID, easting, northing, elevation, length, dip, azimuth |
| surveys | hole ID, depth, dip, azimuth |
| logged intervals (optional) | hole ID, from, to, code — for `validate_contacts.py` |

Column names do not have to match: `core/schema.py` resolves common aliases,
and `describe_mapping()` previews the resolution for any CSV before you run
anything.

```python
import pandas as pd
from core.schema import describe_mapping
print(describe_mapping(pd.read_csv('your_collars.csv'),
                       None,                     # no explicit overrides
                       {'hole': True, 'easting': True, 'northing': True}))
```

It returns one row per internal column — `internal`, `source`, `required`,
`status` — so anything reading `MISSING` tells you what to map by hand via the
`columns` block in `core/config.py`.

Put the files wherever you like — `data/` is gitignored for exactly this — then
point the scripts at them. **The filenames are currently hardcoded** in
`scripts/bakeoff.py`, `validate_contacts.py`, `tune_lsq.py` and `sweep.py`;
edit the constants at the top of each, or see `docs/TODO.md` for the pending
`--data-dir` flag.

Whatever you run, results carrying coordinates or hole IDs
(`out/orientations_*.csv`, `out/contacts.csv`, `out/contact_planes.csv`) are
gitignored by default. Keep it that way — see `docs/TODO.md` §2.

### The bake-off — every estimator, whole property

```bash
.venv/bin/python scripts/bakeoff.py                    # ~13 s
```

Prints the comparison table and writes to `out/`:
`bakeoff_all.png` (stereonets), `bakeoff_summary.csv`, and one
`orientations_all_<estimator>.csv` per estimator (~2 MB each, gitignored).

The number to read is `vs_control_deg` — how far each estimator's poles sit
from the geometry control. Small means it is measuring the drill pattern.

### One prospect, or all of them separately

```bash
.venv/bin/python scripts/bakeoff.py --prospect "Tom West"
.venv/bin/python scripts/bakeoff.py --all-prospects    # ~20 s
```

### Contact validation — the external check

```bash
.venv/bin/python scripts/validate_contacts.py          # ~3 s
```

Fits reference surfaces to the logged geology in `MPA_Interp` and scores the
orientation field against them. Writes `out/contacts.csv`,
`out/contact_planes.csv`, `out/contact_validation.csv`. The first two carry
coordinates and hole IDs and are gitignored; only `contact_validation.csv` is
aggregate-only and tracked.

**This is the only metric that can falsify the field.** Internal metrics
(stereonet tightness, spatial coherence, the estimator's own confidence) all
reward the drilling artifact — see `docs/BAKEOFF.md` §4.

### Tuning sweeps

```bash
.venv/bin/python scripts/tune_lsq.py                   # ridge / conditioning
.venv/bin/python scripts/sweep.py                      # neighbourhood + smoothing
```

### Trying compositing (feature branch only)

Compositing is off unless asked for. In Python:

```python
from core.config import make_cfg
from core.ingest import load_tables
from core.pipeline import prepare, run_estimator

cfg = make_cfg(composite_length_m=5.0)      # None = the raw ~1.3 m intervals
tabs = load_tables('MPA_Samples_BD_20240227.csv',
                   'MPA_Collar_20240227.csv',
                   'MPA_Survey_20240227.csv', cfg)
ds = prepare(*tabs, cfg)
out = run_estimator(ds, cfg, 'lsq_gradient')
```

**Read `docs/COMPOSITING.md` before drawing conclusions from a composited
run.** `min_neighbors` is an absolute sample count, so raising the composite
length silently guts the node set — at 8 m the field collapses to 8 nodes in 3
holes. That is a known limitation with a known fix, not a result.

---

## 6. Where things are

| path | what |
|---|---|
| `core/` | The engine. Pure numpy/scipy/pandas — no plotting, no file I/O, no multiprocessing. |
| `core/config.py` | Every tunable, with the reasoning for each default. Start here. |
| `scripts/` | Runnable comparisons. Deliberately outside `core/`. |
| `viz/` | Stereonets. |
| `tests/` | The 38 (63) tests. |
| `out/` | Figures and result tables. |
| `docs/BAKEOFF.md` | What the bake-off found and why. The main writeup. |
| `docs/COMPOSITING.md` | Compositing, and the `min_neighbors` limitation. Feature branch. |
| `docs/TODO.md` | Standing rules and open work. |

---

## 7. Troubleshooting

**`ModuleNotFoundError: No module named 'core'`** — you are not in the repo
root. `cd` there; the scripts insert the repo root on `sys.path` relative to
their own location, but an interactive session does not.

**`git clone` gives 404** — the repo is private and you are not authenticated.
See §2. A 404 rather than a 403 is GitHub's normal behaviour for a private repo
you cannot see.

**`FileNotFoundError: MPA_Samples_BD_20240227.csv`** — expected on a fresh
clone: the repo ships no data. Supply your own tables per §5.1. If you do have
them, the scripts resolve data paths relative to the working directory, so run
them from the repo root.

**Windows: `.venv/bin/python` is not recognized** — on Windows the interpreter
lives at `.venv\Scripts\python.exe`. See §3.2.

**Windows: "running scripts is disabled on this system"** — PowerShell's
execution policy is blocking the uv installer. The install command in §1.1
already passes `-ExecutionPolicy ByPass` for exactly this reason; if you hit it
elsewhere, `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` lifts it
for the current window only.

**Windows: `py` is not recognized** — the Python launcher is missing, so either
use uv (§3.1, which needs no system Python) or install Python 3.11 with
`winget install --id Python.Python.3.11`.

**pandas or numpy version errors** — the pins in `requirements.txt` are exact.
pandas 3.x changed enough that older 2.x versions are not a safe substitute.

**A bake-off run seems slow** — `--all-prospects` takes about 20 seconds and
prints only as each prospect finishes. The estimators themselves are ~0.1 s
each; the time goes on desurvey and figure rendering.
