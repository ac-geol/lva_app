// Pyodide lives here, off the main thread, so a run never blocks the UI.
//
// The worker owns the engine and nothing else: it boots Pyodide, unpacks
// core/ verbatim from core.zip, loads bridge.py, and forwards messages. Every
// reply is {id, ok, ...}; bulk geometry rides as ArrayBuffers in the transfer
// list rather than being copied through structured clone.

// Served from this bundle, not a CDN: the app has to work on a site network
// with no internet, and that is also the premise that keeps client data on the
// machine. Fetch it once with `python scripts/build_app.py --pyodide`.
const PYODIDE_VERSION = '314.0.7';   // CPython 3.14 series
const PYODIDE_URL = 'pyodide/';

let pyodide = null;
let bridge = null;

function status(stage, message) {
  self.postMessage({ type: 'status', stage, message });
}

async function boot() {
  // Pyodide 314 dropped classic workers, so this is a module worker and the
  // loader arrives by dynamic import rather than importScripts.
  status('loader', `Loading Pyodide ${PYODIDE_VERSION}`);
  const { loadPyodide } = await import(`./${PYODIDE_URL}pyodide.mjs`);

  status('pyodide', `Starting Python ${PYODIDE_VERSION}`);
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  status('packages', 'Loading numpy, pandas, scipy');
  await pyodide.loadPackage(['numpy', 'pandas', 'scipy']);

  status('engine', 'Unpacking the engine');
  const zip = await (await fetch('core.zip')).arrayBuffer();
  pyodide.unpackArchive(zip, 'zip');

  // Fetched rather than zipped so the bridge can be edited without a rebuild.
  const bridgeSource = await (await fetch('bridge.py')).text();
  pyodide.FS.writeFile('bridge.py', bridgeSource);

  bridge = pyodide.pyimport('bridge');

  const versions = call('versions', {});
  self.postMessage({ type: 'ready', versions });
}

// Pyodide hands back a PyProxy; convert once, free it immediately. Python
// bytes arrive as Uint8Array with no per-element conversion.
function call(action, payload) {
  const proxy = bridge.handle(action, JSON.stringify(payload ?? {}));
  try {
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    proxy.destroy();
  }
}

// Anything backed by its own ArrayBuffer can move instead of being copied.
function transferables(value, found = []) {
  if (ArrayBuffer.isView(value)) {
    found.push(value.buffer);
  } else if (Array.isArray(value)) {
    value.forEach((v) => transferables(v, found));
  } else if (value && typeof value === 'object') {
    Object.values(value).forEach((v) => transferables(v, found));
  }
  return found;
}

self.onmessage = async (event) => {
  const { id, action, payload } = event.data;
  try {
    const result = call(action, payload);
    self.postMessage({ id, ...result }, transferables(result));
  } catch (err) {
    // Only harness-level failures reach here; bridge.handle reports engine
    // errors as data so a bad config never looks like a crashed worker.
    self.postMessage({
      id, ok: false, error_type: 'WorkerError',
      error: err && err.message ? err.message : String(err),
    });
  }
};

boot().catch((err) => {
  self.postMessage({
    type: 'boot_error',
    error: err && err.message ? err.message : String(err),
  });
});
