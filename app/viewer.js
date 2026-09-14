// 3D view of the orientation field.
//
// Cheap for one specific reason: the engine already sends the unit normal of
// every estimated plane (pole_vec_x/y/z), so orienting a disc is a single
// quaternion from +Z to that vector. No trigonometry, no angle convention to
// get wrong, and no way for this view and the Leapfrog export to disagree
// about what a plane is.
//
// Coordinates arrive as float32 offsets from a float64 origin. Full UTM values
// exhaust float32's precision and make the whole scene shimmer as the camera
// moves; the offsets keep every value small.

import * as THREE from 'three';
import { OrbitControls } from './three/OrbitControls.js';

const DISC_SEGMENTS = 24;

// Single hue, light to dark, matching the stereonet ramp. Dark-mode steps are
// chosen against the dark surface rather than flipped.
const RAMP_LIGHT = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
const RAMP_DARK  = ['#16304a', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'];

let scene, camera, renderer, controls, clipPlane;
let discs = null, traces = null, host = null;
let data = null, current = {};

const isDark = () => matchMedia('(prefers-color-scheme: dark)').matches;

function rampAt(t) {
  const steps = isDark() ? RAMP_DARK : RAMP_LIGHT;
  const x = Math.max(0, Math.min(1, t)) * (steps.length - 1);
  const i = Math.min(steps.length - 2, Math.floor(x));
  return new THREE.Color(steps[i]).lerp(new THREE.Color(steps[i + 1]), x - i);
}

/** Robust 2nd-98th percentile range, so one outlier cannot flatten the ramp. */
function spread(values) {
  const sorted = Float32Array.from(values).sort();
  const at = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const lo = at(0.02), hi = at(0.98);
  return hi > lo ? [lo, hi] : [lo, lo + 1];
}

export function isReady() { return !!renderer; }

export function init(canvas) {
  if (renderer) return;
  host = canvas;

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.localClippingEnabled = true;

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 1.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(1, 1, 2);
  scene.add(key);

  clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), Infinity);

  const loop = () => { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); };
  loop();
  resize();
  window.addEventListener('resize', resize);
}

function resize() {
  if (!renderer) return;
  const w = host.clientWidth || 720, h = Math.max(380, Math.round(w * 0.62));
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

/** Rebuild the scene from a run's geometry block. */
export function load(geometry) {
  data = {
    n: geometry.n,
    coords: new Float32Array(geometry.coords.buffer),
    poles: new Float32Array(geometry.poles.buffer),
    dip: new Float32Array(geometry.dip.buffer),
    planarity: new Float32Array(geometry.planarity.buffer),
    score: new Float32Array(geometry.score.buffer),
    vsRef: new Float32Array(geometry.vs_reference.buffer),
    traceXYZ: new Float32Array(geometry.trace_xyz.buffer),
    traceOffsets: new Int32Array(geometry.trace_offsets.buffer),
  };

  for (const obj of [discs, traces]) {
    if (obj) { scene.remove(obj); obj.geometry.dispose(); obj.material.dispose(); }
  }

  buildTraces();
  buildDiscs();
  frameAll();
}

function buildTraces() {
  const { traceXYZ, traceOffsets } = data;
  const segments = [];
  for (let h = 0; h < traceOffsets.length - 1; h++) {
    for (let k = traceOffsets[h]; k < traceOffsets[h + 1] - 1; k++) {
      segments.push(traceXYZ[k * 3], traceXYZ[k * 3 + 1], traceXYZ[k * 3 + 2],
                    traceXYZ[(k + 1) * 3], traceXYZ[(k + 1) * 3 + 1], traceXYZ[(k + 1) * 3 + 2]);
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3));
  traces = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({
    color: isDark() ? 0x5d6f7c : 0x9aa8b2, transparent: true, opacity: 0.55,
    clippingPlanes: [clipPlane],
  }));
  scene.add(traces);
}

function buildDiscs() {
  const geom = new THREE.CircleGeometry(1, DISC_SEGMENTS);
  const mat = new THREE.MeshLambertMaterial({
    side: THREE.DoubleSide, transparent: true, opacity: 0.9,
    clippingPlanes: [clipPlane],
  });
  discs = new THREE.InstancedMesh(geom, mat, data.n);
  discs.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(discs);
}

/* Which samples get a disc. Downhole spacing is metres while a legible disc is
   tens of metres across, so drawing every sample stacks them into opaque coins
   and hides the very thing the view exists to show. Thinning is a reading aid,
   not a performance one -- the card says how many are drawn. */
let visible = null;

function setStride(stride) {
  const n = data.n;
  const idx = [];
  for (let k = 0; k < n; k += Math.max(1, stride)) idx.push(k);
  visible = Int32Array.from(idx);
  discs.count = visible.length;
}

/** Point a disc's +Z along the plane's normal, scale it, place it. */
function placeDiscs(radius, scaleByPlanarity) {
  const { coords, poles, planarity } = data;
  const m = new THREE.Matrix4(), q = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 0, 1), normal = new THREE.Vector3();
  const pos = new THREE.Vector3(), scale = new THREE.Vector3();

  for (let i = 0; i < visible.length; i++) {
    const k = visible[i];
    normal.set(poles[k * 3], poles[k * 3 + 1], poles[k * 3 + 2]).normalize();
    q.setFromUnitVectors(up, normal);
    pos.set(coords[k * 3], coords[k * 3 + 1], coords[k * 3 + 2]);
    const s = radius * (scaleByPlanarity ? 0.45 + 0.85 * (planarity[k] || 0) : 1);
    scale.set(s, s, s);
    discs.setMatrixAt(i, m.compose(pos, q, scale));
  }
  discs.instanceMatrix.needsUpdate = true;
}

function colorDiscs(by) {
  const { dip, planarity, score, vsRef } = data;
  const source = { dip, planarity, grade: score, reference: vsRef }[by] || score;

  // Dip and the reference angle are angles on fixed scales; grade and
  // planarity are not, so those get a percentile stretch instead.
  const [lo, hi] = by === 'dip' ? [0, 90]
                 : by === 'reference' ? [0, 60]
                 : spread(source);

  const c = new THREE.Color(), grey = new THREE.Color(0x808080);
  for (let i = 0; i < visible.length; i++) {
    const v = source[visible[i]];
    c.copy(Number.isFinite(v) ? rampAt((v - lo) / (hi - lo)) : grey);
    discs.setColorAt(i, c);
  }
  discs.instanceColor.needsUpdate = true;
}

/** How many discs are actually on screen, for the caption. */
export function shownCount() { return visible ? visible.length : 0; }

export function update(opts) {
  if (!discs || !data) return;
  current = { ...current, ...opts };
  setStride(current.stride ?? 5);
  placeDiscs(current.radius ?? 9, current.scaleByPlanarity !== false);
  colorDiscs(current.colorBy || 'grade');
  traces.visible = current.showTraces !== false;
  discs.scale.set(1, 1, current.exaggeration ?? 1);
  traces.scale.set(1, 1, current.exaggeration ?? 1);

  // Clip along northing, which is how a geologist steps through sections.
  // The plane keeps whatever satisfies normal . p + constant >= 0, so with a
  // normal of -Y that is y <= constant: the cut position goes in directly.
  const bounds = sceneBounds();
  const t = current.section ?? 1;
  clipPlane.normal.set(0, -1, 0);
  clipPlane.constant = bounds.min.y + t * (bounds.max.y - bounds.min.y);
}

function sceneBounds() {
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let k = 0; k < data.n; k++) {
    box.expandByPoint(v.set(data.coords[k * 3], data.coords[k * 3 + 1], data.coords[k * 3 + 2]));
  }
  return box;
}

export function frameAll() {
  const box = sceneBounds();
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z) || 100;

  controls.target.copy(centre);
  camera.position.set(centre.x + span * 0.9, centre.y - span * 1.1, centre.z + span * 0.7);
  camera.up.set(0, 0, 1);                    // Z is elevation, not "up-screen"
  camera.near = span / 1000;
  camera.far = span * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

export function setView(which) {
  const box = sceneBounds();
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z) || 100;

  const offsets = {
    plan:    [0, 0, span * 1.6],
    north:   [0, -span * 1.8, 0],
    east:    [span * 1.8, 0, 0],
    oblique: [span * 0.9, -span * 1.1, span * 0.7],
  };
  const [dx, dy, dz] = offsets[which] || offsets.oblique;
  controls.target.copy(centre);
  camera.position.set(centre.x + dx, centre.y + dy, centre.z + dz);
  controls.update();
}
