# Next session

## 1. Data stripped from the repo — done 2026-08-31

Both goals from the previous session are complete, in the required order:

1. Every data file removed from the working tree **and from git history**.
2. Repo is now *ready* to go public.

**The repo is still private as of this writing.** Flipping it to public is a
deliberate separate step — do it only after the remote has been recreated (see
below), and update `docs/SETUP.md` §1 and §2, which still document the private
clone path.

### What was removed

| file | contained |
|---|---|
| `MPA_*.csv` (4 files, 4.4 MB) | raw collars, surveys, assays, logged intervals |
| `out/contacts.csv` | `HoleID, code, boundary, depth_m, x, y, z` — collar-accurate coordinates |
| `out/contact_planes.csv` | plane fits with centroid coordinates `cx, cy, cz` |
| `lva_notebook_v1.ipynb` / `.md` / `_files/` | see below — the one item the previous audit got wrong |

All still present on disk, just untracked and ignored.

### The notebook: the previous audit was wrong

The previous session recorded that a heuristic scan of the notebook found **0**
hits for hole IDs and coordinates, and flagged it for proper re-verification.
That scan only looked for MacPass-shaped identifiers (`DDH-79-1`, UTM `4xxxxx` /
`70xxxxx`). The notebook is not MacPass — it is a **second, non-public deposit**,
and it carried:

- 10 distinct hole IDs, 60 occurrences in the `.ipynb` and 30 in the `.md`
- 60 full-precision UTM coordinates
- **two georeferenced 3D scatter plots** (`_21_0.png`, `_22_0.png`) with labelled
  Easting/Northing axes — the drill pattern is the deposit footprint
- the project name embedded in ~10 export filenames

MacPass is public data, so none of it was a leak. This was, or would have been.

**Lesson: a scan keyed to one deposit's naming convention proves nothing about
another's.** Grep for the *shape* of a coordinate and an ID, not for known
values, and always look at the figures — axis tick labels are data.

Stereonets (`out/*.png`, and the notebook's `_11_0`, `_25_0`, `_26_0`) were
confirmed by eye to be orientation-only. Safe, as previously assumed.

### How the removal was done

`.gitignore` is now **deny-by-default**: `*.csv` and `data/` ignored outright,
with four explicit `!` exceptions re-admitting the aggregate-only tables
(`bakeoff_summary`, `lsq_tuning`, `structure_tensor_sweep`,
`contact_validation` — angles, counts and timings, no coordinates, no hole IDs).
A new data file is never staged by accident.

History was replaced wholesale via `git checkout --orphan` rather than rewritten
with `filter-repo`: the repo was five commits old with no forks, PRs or external
clones, so a fresh root commit cost nothing and leaves nothing recoverable.

**A force-push would not have been enough.** GitHub keeps unreachable commits
retrievable by direct SHA (`/commit/<sha>`, and via the API) after a force-push,
with no guarantee of prompt garbage collection — so the old blobs would have
become *publicly* retrievable the moment the repo was flipped to public. The
remote must therefore be **deleted and recreated**, not force-pushed.

The 63 tests never touched the data — they still pass on a clone with no tables
in it, which is what makes a public, data-free repo viable rather than a
trade-off.

### Still to do

- **Scripts hardcode the filenames** — `scripts/bakeoff.py`,
  `validate_contacts.py`, `tune_lsq.py`, `sweep.py`. They need a `--data-dir`
  flag or a `data/` convention so each person points at their own tables.
  `docs/SETUP.md` §5.1 documents the manual workaround in the meantime.
- **`docs/BAKEOFF.md` and `README.md` cite MacPass numbers throughout.** Both
  now say explicitly that the numbers are a written record and not reproducible
  from a clean clone. They stay valid as reasoning.

## 2. Standing rule: no data in git

**Nothing containing potentially sensitive data gets committed or pushed on
this project.** Local data only.

This covers derived artifacts, not just raw assay tables — anything carrying
coordinates, hole IDs, or grades. Check what is being staged before `git add
-A`; that command is how data gets in.

---

## 3. Still open from the previous session

Downhole compositing is built and off by default, but **not swept**. Before
sweeping composite length, fix the `min_neighbors` interaction described in
[`docs/COMPOSITING.md`](COMPOSITING.md) §4 — it is an absolute sample count, so
raising the composite length guts the node set and a naive sweep reports a
confident wrong answer. See also README §4.
