# Running the LVA app

A guide for someone who is comfortable following terminal instructions but does
not write Python. You will type about six commands once, then two each time you
want to use the app.

The app runs entirely in your browser. **Your drillhole file is never uploaded
anywhere** — it is read by the page on your own machine, and nothing is sent
over the network once the app is set up.

---

## What you need

| | |
|---|---|
| A terminal | Terminal on macOS, PowerShell on Windows |
| Python | 3.11 is what everything was verified against; 3.12/3.13 very likely fine |
| Access to the repo | It is private — see [`SETUP.md`](SETUP.md) §2 |
| Disk space | About 100 MB, mostly the one-time download |
| A browser | Chrome, Edge, Firefox or Safari, reasonably current |

**No internet is needed to run the app** — only to set it up the first time.
That is deliberate: it means the app works on a site network, and it is why
nothing you load into it can leave your machine.

---

## One-time setup

### 1. Get the code

```bash
gh auth login            # choose HTTPS
gh repo clone ac-geol/lva_app
cd lva_app
```

### 2. Install the Python dependencies

`uv` is the easy route; it fetches the right Python for you.
[`SETUP.md`](SETUP.md) §1.1 covers installing it, and §3.2 covers the plain
`pip` alternative if you prefer.

bash (macOS/Linux):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

PowerShell (Windows):

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

### 3. Download the browser runtime

This is the big one-time step: about 36 MB of the Python runtime and the 3D
library, saved into the app folder so that nothing has to be fetched later.

```bash
.venv/bin/python scripts/build_app.py --pyodide          # macOS/Linux
.venv\Scripts\python.exe scripts\build_app.py --pyodide  # Windows
```

You should see the runtime and six packages listed, ending with a total of
roughly 34 MB, then `three 0.186.0: 2.1 MB`.

### 4. Check it works

```bash
.venv/bin/python -m pytest tests/ -q
```

88 tests should pass. This checks the calculation engine, not the browser — but
if it fails, stop here, because the app runs this same code.

---

## Every time you want to use it

Two steps. From the `lva_app` folder:

```bash
.venv/bin/python scripts/serve_app.py            # macOS/Linux
.venv\Scripts\python.exe scripts\serve_app.py    # Windows
```

It prints a link. Open it in your browser:

```
  http://127.0.0.1:8777/index.html
```

The first load takes 10–20 seconds while Python starts inside the browser — the
badge at the top right says what it is doing. After that it is quick.

**Leave the terminal window open** while you use the app; it is serving the
page. Press `Ctrl+C` in it when you are finished.

---

## Preparing your data in Leapfrog

The app takes **one CSV**, one row per sample interval, already desurveyed. It
needs these columns, under any reasonable name:

| needs | typical names in an export |
|---|---|
| Hole identifier | `HoleID`, `BHID`, `DHID` |
| Interval from / to | `From`, `To`, `Depth_From` |
| Coordinates | `X`, `Y`, `Z` or `Easting`, `Northing`, `Elevation` |
| One assay column | whatever you call it |

The app shows you what it matched and lets you correct any row, so exact names
do not matter.

**Two things worth getting right before you export:**

1. **Filter to the area you care about, then add a margin.** The app looks
   outward from each sample by the search radius (120 m by default) to find its
   neighbours. Samples near the edge of your export only have neighbours on the
   inward side, so their orientations lean toward that cut face — and the
   neighbours beyond the edge are not in the file, so nothing can repair it.
   **Export at least 120 m beyond your area of interest**, and treat the outer
   rim of the result as unreliable. The app measures how much of your file sits
   in that rim and tells you.

2. **Know your interval length.** The minimum-neighbours setting counts
   *samples*, not metres. If your intervals are composited, the same number
   means a different amount of rock. The app shows your median interval and
   what the threshold works out to in metres of core — but it is your call.

---

## Using the app

Six steps, top to bottom. Each unlocks the next.

1. **Data** — drop your CSV on the page, or click *use the bundled example* to
   try it on a synthetic deposit whose correct answer is known.
2. **Columns** — check what it matched. Correct anything wrong from the
   dropdowns.
3. **Assay column** — pick the one element the estimate is built from.
4. **Check** — sample and hole counts, extent, median interval, and the two
   advisories described above. Read them.
5. **Settings** — method, grade cutoff, search radius, minimum neighbours. Each
   tells you what it means for your data as you change it. *Save settings*
   writes a small file so a run can be reproduced later.
6. **Result** — the numbers, two stereonets, a 3D view, and the export.

### Reading the result

The quality numbers are all angles where **about 60° means no signal at all**,
so each is shown against the figure that would count as nothing.

The one to look at first is **distance from the drill pattern**. Every run also
computes what the drilling alone would report, with grades stripped out. If your
answer sits close to that, it may be describing where the holes went rather than
the rock. The second stereonet is that reference, drawn identically for
comparison, and the 3D view can colour every disc by how far it differs.

### The 3D view

Drag to rotate, scroll to zoom. *Plan / Look north / Look east* jump to standard
views; *Section* slices through; *Vertical ×* exaggerates elevation. Discs are
thinned to 1 in 5 by default because at metre sample spacing they otherwise
stack into solid coins.

Colouring by **difference from drill pattern** is the check worth doing: dark
discs are where the answer is following the drilling rather than the
mineralisation. A stereonet cannot show you that.

### The export

*Download Leapfrog CSV* writes dip, dip azimuth and pitch, along with the
coordinates, confidence measures and the drill-pattern comparison for each row.
The panel above the button explains every column.

> **One caution.** Dip and dip azimuth are verified against Leapfrog. **Pitch
> is not yet** — the direction it is measured in has not been confirmed, and a
> reversed pitch gives an ellipsoid correctly oriented as a plane but with its
> long axis pointing the wrong way inside it. [`LEAPFROG_CHECK.md`](LEAPFROG_CHECK.md)
> is a half-hour procedure that settles it. Until then, treat exported pitch as
> provisional.

---

## If something goes wrong

**The page says "Failed to start" or sits on "Starting…".**
The runtime was not downloaded. Run step 3 of the setup again.

**The browser shows "This site can't be reached".**
The terminal command is not running, or it is in the wrong folder. It must be
run from inside `lva_app`.

**"Address already in use".**
An earlier session is still running. Close that terminal window, or use a
different port: `scripts/serve_app.py --port 8800`.

**Some required column says MISSING.**
Pick the right column from the dropdown next to it. If your file genuinely has
no coordinates, it has not been desurveyed — re-export it with X/Y/Z.

**"No orientations could be estimated."**
Not a crash. There were not enough neighbouring samples to work with, and the
message says which limit was hit and what your data actually offers. Widen the
search radius or lower the minimum neighbours.

**Nothing appears in the 3D view.**
Your browser may not support WebGL, or it is disabled. The stereonets and the
export still work.

**You changed the code and the app behaves as if you had not.**
Always start the app with `scripts/serve_app.py`, not a plain web server — it
rebuilds the engine snapshot the browser loads. That is the whole reason it
exists.

---

## Sharing results with colleagues

The exported CSV and the saved settings file are ordinary files; send them as
you would any others.

Sharing the *app* is harder today: a browser will not run it from a folder on
disk, so each person needs this same setup. Putting the built `app/` folder on
an internal web server would remove that — everyone would just open a link, and
their data would still never leave their own machine — but that is a separate
piece of work.
