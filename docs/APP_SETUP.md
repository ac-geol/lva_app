# Setting up and using the LVA app

This guide assumes you have never used Python and have barely used a terminal.
Every command is written out in full for both Mac and Windows — you copy and
paste, you do not need to understand them.

**Setting up takes about fifteen minutes and happens once.** After that,
starting the app is one command.

**Your drillhole data never leaves your computer.** The app runs inside your own
browser: the file you load is read on your machine, nothing is uploaded, and
once set up the app needs no internet at all. That is deliberate, and it is why
it works on a locked-down site network.

---

## A few words you will meet

| word | what it means here |
|---|---|
| **terminal** | A window where you type commands instead of clicking. Called Terminal on a Mac, PowerShell on Windows. |
| **repo** | The folder of code you are about to download, and its page on GitHub. |
| **uv** | A small tool that installs Python for you. It saves you having to install Python yourself. |
| **virtual environment** (`.venv`) | A private copy of Python that lives inside this project folder, so nothing you do here can disturb anything else on your computer. |
| **runtime** | The Python that runs *inside the browser*. It is a one-time 36 MB download. |

---

## What you need

| | |
|---|---|
| A computer | Mac, Windows or Linux. |
| Disk space | About 200 MB. |
| A browser | Chrome, Edge, Firefox or Safari, reasonably current. |
| Internet | For the setup only. Not to use the app afterwards. |

You do **not** need Python installed already — step 2 takes care of it.

---

## Opening a terminal

**On a Mac:** press `Cmd + Space`, type `terminal`, press Enter.

**On Windows:** press the Start button, type `powershell`, press Enter.

A window opens with a blinking cursor. Commands go there: paste one, press
Enter, wait for the cursor to come back before pasting the next.

Two things worth knowing:

- **Pasting:** `Cmd + V` on a Mac, `Ctrl + V` or a right-click on Windows.
- **Nothing happening is normal.** Some commands take a minute and print
  nothing while they work. Wait for the cursor to reappear.

---

# One-time setup

## Step 1 — Get the code

The repo is public, so this needs no account and no password.

**The simple way — download it:**

1. Go to <https://github.com/ac-geol/lva_app>
2. Click the green **Code** button, then **Download ZIP**.
3. Unzip it. You will get a folder called `lva_app-main`.
4. Move that folder somewhere you can find it again — your Desktop is fine.

**Then tell the terminal to work inside that folder.** Type `cd` and a space,
then drag the folder from Finder or File Explorer onto the terminal window and
press Enter. That fills in the path for you:

```
cd /Users/you/Desktop/lva_app-main
```

**The alternative, if you have `git`:** this makes updating later a one-line
`git pull` instead of another download.

```
git clone https://github.com/ac-geol/lva_app.git
cd lva_app
```

> **Every command from here on must be run inside that folder.** If you close
> the terminal and open a new one, do the `cd` step again first.

## Step 2 — Install uv

This installs the helper that fetches Python for you. One line.

**Mac / Linux:**

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Now close the terminal window and open a new one**, then `cd` back into the
project folder. This is not optional — the new window is what knows `uv`
exists.

Check it worked by typing `uv --version`. You should see a version number. If
you see "command not found", the window was not reopened.

## Step 3 — Build the private Python environment

Two commands. The first makes the environment, the second installs the five
libraries the engine uses. Expect a minute or two.

**Mac / Linux:**

```
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

**Windows:**

```
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

**What success looks like:** a list of downloaded packages ending in a line
like `Installed 21 packages`. A new `.venv` folder has appeared inside the
project folder.

## Step 4 — Download the browser runtime

This is the big one: about 36 MB of Python and the 3D library, saved into the
project so the app never has to fetch anything again.

**Mac / Linux:**

```
.venv/bin/python scripts/build_app.py --pyodide
```

**Windows:**

```
.venv\Scripts\python.exe scripts\build_app.py --pyodide
```

**What success looks like:** a few lines listing the runtime and six packages,
totalling roughly 34 MB, then a last line reading `three 0.186.0:  2.1 MB`.

## Step 5 — Check that it works

**Mac / Linux:**

```
.venv/bin/python -m pytest tests/ -q
```

**Windows:**

```
.venv\Scripts\python.exe -m pytest tests\ -q
```

**What success looks like:** a row of dots, then `62 passed`.

This tests the calculation engine, not the browser — but if it fails, stop
here and do not use the app, because the app runs this same code.

Setup is done.

---

# Using the app

## Starting it

One command, from the project folder.

**Mac / Linux:**

```
.venv/bin/python scripts/serve_app.py
```

**Windows:**

```
.venv\Scripts\python.exe scripts\serve_app.py
```

It prints a link:

```
  http://127.0.0.1:8777/index.html
```

Open that in your browser — click it if your terminal makes it clickable,
otherwise copy and paste it into the address bar.

The first load takes 10–20 seconds while Python starts up inside the browser.
The badge at the top right says what it is doing. After that it is quick.

> **Leave the terminal window open** while you use the app — it is what serves
> the page. When you are finished, click the terminal and press `Ctrl + C`.

## Preparing your data

The app takes **one CSV file: one row per sample interval, already
desurveyed** — meaning each row already carries its real-world X, Y and Z
coordinates. That is what Leapfrog, Datamine and their equivalents export when
you export points or composites. There are no collar or survey files to supply:
the app does no desurveying and asks for none.

It needs these columns, under any reasonable name:

| it needs | names it will recognise |
|---|---|
| Hole identifier | `HoleID`, `BHID`, `DHID` |
| Interval from / to | `From`, `To`, `Depth_From` |
| Coordinates | `X`, `Y`, `Z`, or `Easting`, `Northing`, `Elevation` |
| One assay column | whatever you call it |

The app shows you what it matched and lets you correct any row, so exact names
do not matter.

**Two things worth getting right before you export:**

1. **Export a margin around your area of interest.** The app looks outward from
   each sample by the search radius — 120 m by default — to find its
   neighbours. Samples near the edge of your export only have neighbours on the
   inward side, so their orientations lean toward that cut face, and the
   neighbours beyond the edge are not in the file, so nothing can repair it.
   **Export at least 120 m beyond the area you care about**, and treat the
   outer rim of the result as unreliable. The app measures how much of your
   file sits in that rim and tells you.

2. **Know your interval length.** The minimum-neighbours setting counts
   *samples*, not metres. Composited data means the same number stands for a
   different amount of rock. The app shows your median interval and what the
   threshold works out to in metres of core — but the judgement is yours.

## The six steps on screen

Top to bottom. Each unlocks the next.

1. **Data** — drag your CSV onto the page, or click *use the bundled example*
   to try the app on a synthetic deposit whose correct answer is known.
2. **Columns** — check what it matched. Fix anything wrong from the dropdowns.
3. **Assay column** — pick the one element the estimate is built from.
4. **Check** — sample and hole counts, extent, median interval, and the two
   advisories above. Read them.
5. **Settings** — method, grade cutoff, search radius, minimum neighbours. Each
   explains what it means for your data as you change it. *Save settings*
   writes a small file so a run can be reproduced later.
6. **Result** — the numbers, two stereonets, a 3D view, and the export.

## Reading the result

The quality numbers are all angles, and **about 60° means no signal at all** —
so each is shown next to the figure that would count as nothing.

The one to look at first is **distance from the drill pattern**. Every run also
works out what the drilling alone would report, with the grades stripped out.
If your answer sits close to that, it may be describing where the holes went
rather than the rock. The second stereonet is that reference, drawn identically
so the two can be compared by eye.

## The 3D view

Drag to rotate, scroll to zoom. *Plan / Look north / Look east* jump to standard
views, *Section* slices through, and *Vertical ×* exaggerates elevation. Discs
are thinned to 1 in 5 by default, because at metre sample spacing they otherwise
stack into solid coins.

Colouring by **difference from drill pattern** is the check worth doing: dark
discs are where the answer is following the drilling rather than the
mineralisation. A stereonet cannot show you that.

## The export

*Download Leapfrog CSV* writes dip, dip azimuth and pitch, along with the
coordinates, the confidence measures and the drill-pattern comparison for every
row. The panel above the button explains each column.

> **One caution.** Dip and dip azimuth are verified against Leapfrog. **Pitch
> is not yet** — the direction it is measured in has not been confirmed, and a
> reversed pitch gives an ellipsoid correctly oriented as a plane but with its
> long axis pointing the wrong way inside it.
> [`LEAPFROG_CHECK.md`](LEAPFROG_CHECK.md) is a half-hour procedure that
> settles it. Until then, treat exported pitch as provisional.

---

## If something goes wrong

**The page says "Failed to start", or sits on "Starting…".**
The browser runtime was not downloaded. Run step 4 again.

**The browser says "This site can't be reached".**
The terminal command is not running, or it was run from the wrong place. It
must be run from inside the project folder — repeat the `cd` step, then start
it again.

**"Address already in use".**
The app is already running in another terminal window. Use that one, or start
this on a different port by adding `--port 8800` to the end of the command.

**A required column says MISSING.**
Pick the right column from the dropdown beside it. If your file genuinely has
no X/Y/Z coordinates, it has not been desurveyed — re-export it with them.

**"No orientations could be estimated."**
Not a crash. There were not enough neighbouring samples to work with. The
message says which limit was hit and what your data actually offers: widen the
search radius, or lower the minimum neighbours.

**Nothing appears in the 3D view.**
Your browser may not support WebGL, or it is switched off. The stereonets and
the export still work.

**`command not found: uv`.**
The terminal was not reopened after step 2. Close it, open a new one, `cd` back
into the project folder.

**Windows: "running scripts is disabled on this system".**
PowerShell is blocking the installer. The command in step 2 already works
around this; if you hit it elsewhere, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first — it applies
to that window only.

**You edited the code and the app behaves as though you had not.**
Always start the app with `scripts/serve_app.py` rather than a plain web
server. It rebuilds the engine snapshot the browser loads, which is the whole
reason it exists.

---

## Sharing results

The exported CSV and the saved settings file are ordinary files — send them as
you would any others.

Sharing the *app* is harder today: a browser will not run it from a folder on
disk, so each person needs this same setup. Putting the built `app/` folder on
an internal web server would remove that — everyone would just open a link, and
their data would still never leave their own machine — but that is a separate
piece of work.
