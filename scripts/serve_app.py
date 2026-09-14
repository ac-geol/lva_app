"""Rebuild the app's assets, then serve them.

`core.zip` is a snapshot of `core/`, so editing the engine and reloading the
page silently serves the old code -- the browser shows stale defaults and
stale behaviour with nothing to say so. Rebuilding on every start removes that
trap; use this rather than pointing a plain http.server at app/.

    python scripts/serve_app.py [--port 8777]
"""
from __future__ import annotations

import argparse
import functools
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_app  # noqa: E402  - sibling script, imported after sys.path is set


class Handler(http.server.SimpleHTTPRequestHandler):
    """No-cache, so a rebuilt core.zip is actually picked up on reload."""

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

    def log_message(self, fmt, *args):
        if '404' in (fmt % args):
            super().log_message(fmt, *args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8777)
    args = ap.parse_args()

    build_app.APP.mkdir(exist_ok=True)
    print(f"core.zip  {build_app.build_core_zip()} modules", flush=True)
    if not build_app.SAMPLE_CSV.exists():
        print(f"sample    {build_app.build_sample_csv()} rows", flush=True)
    if not (build_app.PYODIDE_DIR / 'pyodide.mjs').exists():
        sys.exit("Pyodide is not vendored. Run: python scripts/build_app.py --pyodide")

    # Threading matters: the page and the worker fetch concurrently (the CSV
    # while Pyodide pulls wheels), and a single-threaded server serialises them
    # into what looks like a hang.
    handler = functools.partial(Handler, directory=str(build_app.APP))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler) as httpd:
        print(f"\n  http://127.0.0.1:{args.port}/index.html\n", flush=True)
        httpd.serve_forever()


if __name__ == '__main__':
    main()
