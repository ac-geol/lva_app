"""Build the browser app's assets.

Three outputs, all build artifacts rather than source:

  core.zip     `core/` verbatim, so the browser runs the same modules the tests
               cover -- there is no second copy of the engine to keep in step.
  sample CSV   generated from `core/synthetic.py` rather than committed, which
               keeps the repo free of drillhole data by construction while
               still giving the app something to open.
  pyodide/     the Python runtime and the three wheels the engine needs.

Pyodide is vendored rather than loaded from a CDN on purpose. The whole point
of the browser build is that it runs with no server and no data leaving the
machine (README section 5); a runtime CDN dependency would quietly break that
for anyone on a site network with no internet, and it is exactly what a
locked-down browser blocks first.

    python scripts/build_app.py                # engine + sample
    python scripts/build_app.py --pyodide      # also fetch the runtime (~60 MB)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = ROOT / 'app'
CORE_ZIP = APP / 'core.zip'
SAMPLE_CSV = APP / 'sample_synthetic.csv'
PYODIDE_DIR = APP / 'pyodide'

# Pinned. Pyodide tracks CPython, so 314.x is the Python 3.14 series.
PYODIDE_VERSION = '314.0.7'
PYODIDE_CDN = f'https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/'

# The ESM build and its companions. Pyodide 314 dropped classic workers, so
# the worker is a module worker importing pyodide.mjs -- which in turn pulls
# pyodide.asm.mjs, a name the loader builds at runtime rather than spelling out
# in its source, so it has to be listed here explicitly.
RUNTIME_FILES = ['pyodide.mjs', 'pyodide.asm.mjs', 'pyodide.asm.wasm',
                 'python_stdlib.zip', 'pyodide-lock.json']

# Wheels are resolved from the lock file, so transitive dependencies
# (python-dateutil, pytz, six) come along without being listed here.
WHEELS_FOR = ['numpy', 'pandas', 'scipy']

# three.js for the 3D view, vendored for the same reason Pyodide is: the app
# has to run with no internet. three.module.js pulls three.core.js by relative
# path; OrbitControls imports the bare specifier 'three', which index.html
# resolves with an import map.
THREE_VERSION = '0.186.0'
THREE_CDN = f'https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/'
THREE_FILES = {
    'build/three.module.js': 'three.module.js',
    'build/three.core.js': 'three.core.js',
    'examples/jsm/controls/OrbitControls.js': 'OrbitControls.js',
}


def build_core_zip() -> int:
    """Zip `core/` with its package path intact, so `import core` just works."""
    sources = sorted(p for p in (ROOT / 'core').glob('*.py'))
    with zipfile.ZipFile(CORE_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, arcname=f'core/{path.name}')
    return len(sources)


def build_sample_csv() -> int:
    """A desurveyed point table in the shape Leapfrog exports.

    Uses the project's own minimum-curvature desurvey to place the samples, so
    the file is exactly what the app expects a user to bring: hole, interval,
    one assay column, and X/Y/Z. The ground-truth columns ride along as extra
    columns the app ignores, which lets the browser check its own answer.
    """
    from core.ingest import load_tables
    from core.pipeline import prepare
    from core.schema import COLS
    from core.synthetic import example_cfg, make_example_deposit

    cfg = example_cfg(score_columns=['Cu_pct'])
    dep = make_example_deposit(seed=0)
    ds = prepare(*load_tables(dep.samples, dep.collars, dep.surveys, cfg), cfg)

    truth = ['true_pole_x', 'true_pole_y', 'true_pole_z']
    out = ds.des[[COLS['hole'], COLS['from'], COLS['to'], 'Cu_pct', *truth]].copy()
    out['X'] = ds.des['mid_x'].to_numpy()
    out['Y'] = ds.des['mid_y'].to_numpy()
    out['Z'] = ds.des['mid_z'].to_numpy()

    out = out.rename(columns={COLS['hole']: 'HoleID',
                              COLS['from']: 'From', COLS['to']: 'To'})
    # 10 significant digits: enough that the browser's answer can be
    # compared to the CLI's without rounding muddying the comparison.
    out.to_csv(SAMPLE_CSV, index=False, float_format='%.10g')
    return len(out)


def _fetch(name: str, dest: Path) -> int:
    """Download one distribution file, skipping it if already present."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(PYODIDE_CDN + name, timeout=180) as r:
        data = r.read()
    dest.write_bytes(data)
    return len(data)


def vendor_pyodide() -> int:
    """Fetch the runtime and only the wheels the engine actually imports."""
    total = sum(_fetch(name, PYODIDE_DIR / name) for name in RUNTIME_FILES)

    lock = json.loads((PYODIDE_DIR / 'pyodide-lock.json').read_text())
    packages = {v['name']: v for v in lock['packages'].values()}

    wanted, queue = set(), list(WHEELS_FOR)
    while queue:
        name = queue.pop()
        if name in wanted or name not in packages:
            continue
        wanted.add(name)
        queue.extend(packages[name].get('depends', []))

    for name in sorted(wanted):
        file_name = packages[name]['file_name']
        size = _fetch(file_name, PYODIDE_DIR / file_name)
        print(f"  {name:18s} {packages[name]['version']:12s} {size / 1048576:6.1f} MB")
        total += size

    return total


def vendor_three() -> int:
    """Fetch the three.js modules the viewer imports."""
    dest_dir = APP / 'three'
    total = 0
    for remote, local in THREE_FILES.items():
        dest = dest_dir / local
        if dest.exists() and dest.stat().st_size > 0:
            total += dest.stat().st_size
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(THREE_CDN + remote, timeout=180) as r:
            data = r.read()
        dest.write_bytes(data)
        total += len(data)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pyodide', action='store_true',
                    help='also download the Pyodide runtime and wheels')
    args = ap.parse_args()

    APP.mkdir(exist_ok=True)
    n_modules = build_core_zip()
    n_rows = build_sample_csv()
    print(f"core.zip             {n_modules} modules   "
          f"{CORE_ZIP.stat().st_size / 1024:.1f} kB")
    print(f"sample_synthetic.csv {n_rows} rows       "
          f"{SAMPLE_CSV.stat().st_size / 1024:.1f} kB")

    if args.pyodide:
        print(f"pyodide {PYODIDE_VERSION}:")
        total = vendor_pyodide()
        print(f"  -> {total / 1048576:.1f} MB in {PYODIDE_DIR.relative_to(ROOT)}/")
        print(f"three {THREE_VERSION}:  {vendor_three() / 1048576:.1f} MB")
    else:
        missing = []
        if not (PYODIDE_DIR / 'pyodide.mjs').exists():
            missing.append('Pyodide')
        if not (APP / 'three' / 'three.module.js').exists():
            missing.append('three.js')
        if missing:
            print(f"\n{' and '.join(missing)} not vendored yet. Run:")
            print("    python scripts/build_app.py --pyodide")


if __name__ == '__main__':
    main()
