"""The eigenvector-assignment guard.

Each estimator declares which eigenvector carries the pole. These tests build
synthetic fields whose correct answer is known by construction and assert the
estimators recover it -- the only defence against a result that is 90 degrees
wrong and still looks plausible on a stereonet.
"""
import numpy as np
import pytest

from core.config import make_cfg
from core.estimators import (STRUCTURE_TENSOR_LINEARITY_CEILING,
                             STRUCTURE_TENSOR_PLANARITY_CEILING, TensorField,
                             edge_tensor, field_to_orientations, shape_pca,
                             structure_tensor)
from core.geometry import angular_difference, orientation_tensor, unitize
from core.neighbors import EdgeSet, build_edge_set

TRUE_POLE = unitize(np.array([np.sin(np.radians(35)) * np.sin(np.radians(120)),
                              np.sin(np.radians(35)) * np.cos(np.radians(120)),
                              np.cos(np.radians(35))]))


def _grid(n=12, spacing=15.0, seed=0):
    """A regular 3D block of points -- deliberately NOT drillhole-shaped, so
    these tests isolate estimator maths from sampling geometry."""
    rng = np.random.default_rng(seed)
    g = np.arange(n) * spacing
    pts = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    pts = pts + rng.normal(scale=0.5, size=pts.shape)
    # Scatter across many synthetic holes so min_holes / declustering are met.
    hole_code = rng.integers(0, 40, size=len(pts)).astype(np.int64)
    return pts, hole_code


def _cfg(**kw):
    base = dict(radius_m=40.0, min_neighbors=8, min_holes=3,
                distance_sigma_m=30.0, exclude_same_hole_neighbors=True,
                decluster_by_hole=True)
    base.update(kw)
    return make_cfg(**base)


# --------------------------------------------------------------------------
# The inversion itself, on a tensor whose eigenvectors are known exactly.
# --------------------------------------------------------------------------

def _diag_field(pole_eigenvector, major_axis, ratio=(4.0, 1.0, 1.0)):
    """One node whose tensor has `major_axis` as its dominant eigenvector."""
    a = unitize(np.asarray(major_axis, float))
    tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b = unitize(np.cross(a, tmp))
    c = np.cross(a, b)
    R = np.stack([a, b, c], axis=1)                 # columns = eigenvectors
    T = R @ np.diag(ratio) @ R.T
    return TensorField(T=T[None], pole_eigenvector=pole_eigenvector,
                       valid=np.array([True]), name='synthetic'), a, b, c


def test_major_convention_puts_pole_on_the_major_axis():
    tf, a, b, c = _diag_field('major', TRUE_POLE)
    out = field_to_orientations(tf)
    pole = out[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
    line = out[['line_vec_x', 'line_vec_y', 'line_vec_z']].to_numpy()
    assert angular_difference(pole, a)[0] < 1e-6
    assert angular_difference(line, a)[0] == pytest.approx(90.0, abs=1e-6)


def test_minor_convention_puts_pole_on_the_minor_axis():
    tf, a, b, c = _diag_field('minor', TRUE_POLE)
    out = field_to_orientations(tf)
    pole = out[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
    line = out[['line_vec_x', 'line_vec_y', 'line_vec_z']].to_numpy()
    assert angular_difference(line, a)[0] < 1e-6           # major = lineation
    assert angular_difference(pole, a)[0] == pytest.approx(90.0, abs=1e-6)


def test_the_two_conventions_are_orthogonal_on_the_same_tensor():
    """Same tensor, opposite declarations -> poles 90 degrees apart.

    This is precisely the failure this module exists to prevent.
    """
    major, _, _, _ = _diag_field('major', TRUE_POLE)
    minor = TensorField(T=major.T, pole_eigenvector='minor',
                        valid=major.valid, name='synthetic')
    pa = field_to_orientations(major)[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
    pb = field_to_orientations(minor)[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
    assert angular_difference(pa, pb)[0] == pytest.approx(90.0, abs=1e-6)


def test_planarity_is_defined_by_meaning_not_eigenvalue_position():
    """(4,1,1) is one dominant axis: planar under 'major', linear under 'minor'."""
    major, *_ = _diag_field('major', TRUE_POLE, ratio=(4.0, 1.0, 1.0))
    minor = TensorField(T=major.T, pole_eigenvector='minor',
                        valid=major.valid, name='synthetic')
    a = field_to_orientations(major).iloc[0]
    b = field_to_orientations(minor).iloc[0]
    assert a['planarity'] == pytest.approx(0.75)
    assert a['linearity'] == pytest.approx(0.0)
    assert b['linearity'] == pytest.approx(0.75)
    assert b['planarity'] == pytest.approx(0.0)


def test_declared_convention_is_validated():
    with pytest.raises(ValueError):
        TensorField(T=np.eye(3)[None], pole_eigenvector='biggest',
                    valid=np.array([True]), name='bad')


# --------------------------------------------------------------------------
# Each estimator against a synthetic field with a known answer.
# --------------------------------------------------------------------------

def test_structure_tensor_recovers_the_pole_of_a_layered_grade_field():
    """Grade varies ONLY along TRUE_POLE, so the gradient is TRUE_POLE.

    Wavelength is kept long relative to the search radius: a finite-difference
    estimator aliases once the field turns over inside one neighbourhood.
    """
    coords, hole_code = _grid()
    scores = np.sin(2 * np.pi * (coords @ TRUE_POLE) / 200.0)
    lengths = np.full(len(coords), 2.0)
    cfg = _cfg()

    edges = build_edge_set(coords, scores, lengths, hole_code, cfg)
    tf = structure_tensor(edges, coords, scores, hole_code, cfg)
    out = field_to_orientations(tf)
    out = out.loc[out['valid']]

    poles = out[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
    err = angular_difference(poles, np.broadcast_to(TRUE_POLE, poles.shape))

    # Per-node scatter is limited by neighbour-count noise; the aggregate is not.
    assert np.median(err) < 12.0
    _, evecs = orientation_tensor(poles)
    assert angular_difference(evecs[:, 0], TRUE_POLE)[0] < 3.0
    assert out['planarity_norm'].median() > 0.85


def test_structure_tensor_planarity_ceiling_is_two_thirds():
    """A perfect plane cannot push this estimator's raw planarity above 2/3.

    Isotropic neighbours + a linear grade field give T = diag(1/5, 1/15, 1/15)
    analytically. Comparing this raw against shape-PCA (which reaches 1.0) is
    the mistake the _norm columns exist to prevent.
    """
    rng = np.random.default_rng(11)
    u = unitize(rng.normal(size=(120000, 3)))
    grad = np.array([0.0, 0.0, 1.0])
    T = np.einsum('n,ni,nj->ij', (u @ grad) ** 2, u, u) / len(u)
    ev = np.sort(np.linalg.eigvalsh(T))[::-1]
    assert ev == pytest.approx([1 / 5, 1 / 15, 1 / 15], abs=5e-3)
    assert (ev[0] - ev[1]) / ev[0] == pytest.approx(
        STRUCTURE_TENSOR_PLANARITY_CEILING, abs=0.02)


def test_structure_tensor_linearity_ceiling_is_one_half():
    """A perfect rod: the gradient rotates within a plane across the ball."""
    rng = np.random.default_rng(12)
    th = rng.uniform(0, 2 * np.pi, 120000)
    m = np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
    T = (1 / 15) * np.eye(3) + (2 / 15) * np.einsum('ni,nj->ij', m, m) / len(m)
    ev = np.sort(np.linalg.eigvalsh(T))[::-1]
    assert (ev[1] - ev[2]) / ev[0] == pytest.approx(
        STRUCTURE_TENSOR_LINEARITY_CEILING, abs=0.02)
    assert (ev[0] - ev[1]) / ev[0] == pytest.approx(0.0, abs=0.02)


def test_normalized_metrics_are_on_a_common_scale():
    """Two estimators, same perfect structure -> comparable _norm values."""
    tf, a, _, _ = _diag_field('major', TRUE_POLE, ratio=(1 / 5, 1 / 15, 1 / 15))
    tf.planarity_ceiling = STRUCTURE_TENSOR_PLANARITY_CEILING
    tf.linearity_ceiling = STRUCTURE_TENSOR_LINEARITY_CEILING
    st = field_to_orientations(tf).iloc[0]

    pca, *_ = _diag_field('minor', TRUE_POLE, ratio=(1.0, 1.0, 0.0))
    sp = field_to_orientations(pca).iloc[0]

    assert st['planarity'] == pytest.approx(2 / 3, abs=1e-6)     # raw differs
    assert sp['planarity'] == pytest.approx(1.0, abs=1e-6)
    assert st['planarity_norm'] == pytest.approx(1.0, abs=1e-6)  # normalized agrees
    assert sp['planarity_norm'] == pytest.approx(1.0, abs=1e-6)


def test_structure_tensor_reports_no_structure_in_a_uniform_field():
    """Constant grade -> zero gradient energy -> nothing to orient."""
    coords, hole_code = _grid()
    scores = np.full(len(coords), 1.5)
    lengths = np.full(len(coords), 2.0)
    cfg = _cfg()

    edges = build_edge_set(coords, scores, lengths, hole_code, cfg)
    tf = structure_tensor(edges, coords, scores, hole_code, cfg)
    assert np.nanmax(tf.aux['gradient_energy']) < 1e-12


def test_shape_pca_recovers_the_normal_of_a_planar_point_cloud():
    """Points confined to a plane -> the short axis is that plane's pole."""
    rng = np.random.default_rng(3)
    n = 4000
    b = unitize(np.cross(TRUE_POLE, [0.0, 0.0, 1.0]))
    c = np.cross(TRUE_POLE, b)
    uv = rng.uniform(-100, 100, size=(n, 2))
    coords = (uv[:, :1] * b + uv[:, 1:] * c
              + rng.normal(scale=1.0, size=(n, 1)) * TRUE_POLE)
    hole_code = rng.integers(0, 40, size=n).astype(np.int64)
    scores = np.ones(n)
    lengths = np.full(n, 2.0)
    cfg = _cfg(radius_m=30.0, min_neighbors=15)

    edges = build_edge_set(coords, scores, lengths, hole_code, cfg)
    tf = shape_pca(edges, coords, scores, hole_code, cfg)
    out = field_to_orientations(tf)
    out = out.loc[out['valid']]

    poles = out[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
    err = angular_difference(poles, np.broadcast_to(TRUE_POLE, poles.shape))
    assert np.median(err) < 10.0


def test_edge_tensor_recovers_a_lineation():
    """Grade constant along Z, varying in X and Y -> a vertical rod."""
    coords, hole_code = _grid()
    axis = np.array([0.0, 0.0, 1.0])
    scores = (np.sin(2 * np.pi * coords[:, 0] / 50.0)
              * np.sin(2 * np.pi * coords[:, 1] / 50.0))
    lengths = np.full(len(coords), 2.0)
    cfg = _cfg(grade_sim_sigma=0.05)

    edges = build_edge_set(coords, scores, lengths, hole_code, cfg)
    tf = edge_tensor(edges, coords, scores, hole_code, cfg)
    out = field_to_orientations(tf)
    out = out.loc[out['valid']]

    lines = out[['line_vec_x', 'line_vec_y', 'line_vec_z']].to_numpy()
    err = angular_difference(lines, np.broadcast_to(axis, lines.shape))

    # Nodes sitting on a nodal line of the grade field have no local gradient
    # and legitimately return noise, so judge the population, not every node.
    assert np.median(err) < 30.0
    _, evecs = orientation_tensor(lines)
    assert angular_difference(evecs[:, 0], axis)[0] < 5.0


# --------------------------------------------------------------------------
# Least-squares gradient reconstruction.
# --------------------------------------------------------------------------

def test_lsq_gradient_recovers_the_pole_of_a_layered_field():
    from core.estimators import lsq_gradient

    coords, hole_code = _grid()
    scores = np.sin(2 * np.pi * (coords @ TRUE_POLE) / 200.0)
    lengths = np.full(len(coords), 2.0)
    cfg = _cfg()

    edges = build_edge_set(coords, scores, lengths, hole_code, cfg)
    out = field_to_orientations(
        lsq_gradient(edges, coords, scores, hole_code, cfg))
    out = out.loc[out['valid']]

    poles = out[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
    err = angular_difference(poles, np.broadcast_to(TRUE_POLE, poles.shape))
    assert np.median(err) < 6.0
    assert out['gradient_r2'].median() > 0.8


def test_lsq_gradient_beats_the_structure_tensor_on_anisotropic_sampling():
    """The whole reason this estimator exists.

    Sampling is restricted to a narrow cone of directions -- the drillhole
    situation. Averaging u u^T inherits that bias; solving for the gradient
    inverts it away.
    """
    from core.estimators import lsq_gradient

    rng = np.random.default_rng(21)
    n = 5000
    # Points spread widely in x and y but thinly in z: neighbour directions are
    # therefore strongly confined to the horizontal plane.
    coords = np.column_stack([rng.uniform(0, 400, n), rng.uniform(0, 400, n),
                              rng.normal(0, 12, n)])
    hole_code = rng.integers(0, 60, n).astype(np.int64)
    # ...but the grade varies almost entirely along z, the poorly sampled axis.
    true_pole = unitize(np.array([0.15, 0.1, 1.0]))
    scores = (coords @ true_pole) / 100.0
    lengths = np.full(n, 2.0)
    cfg = _cfg(radius_m=60.0, min_neighbors=20)

    edges = build_edge_set(coords, scores, lengths, hole_code, cfg)

    def med_err(tf):
        o = field_to_orientations(tf)
        o = o.loc[o['valid']]
        p = o[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
        return float(np.median(
            angular_difference(p, np.broadcast_to(true_pole, p.shape))))

    err_st = med_err(structure_tensor(edges, coords, scores, hole_code, cfg))
    err_ls = med_err(lsq_gradient(edges, coords, scores, hole_code, cfg))

    assert err_ls < 5.0
    assert err_ls < err_st / 2.0


def test_lsq_gradient_reports_degenerate_sampling():
    """Coplanar neighbours cannot constrain the out-of-plane gradient, and the
    conditioning number must say so rather than returning false confidence."""
    from core.estimators import lsq_gradient

    rng = np.random.default_rng(22)
    n = 3000
    coords = np.column_stack([rng.uniform(0, 400, n), rng.uniform(0, 400, n),
                              np.zeros(n)])                    # perfectly flat
    hole_code = rng.integers(0, 60, n).astype(np.int64)
    scores = coords[:, 0] / 100.0
    lengths = np.full(n, 2.0)
    cfg = _cfg(radius_m=60.0, min_neighbors=20)

    edges = build_edge_set(coords, scores, lengths, hole_code, cfg)
    tf = lsq_gradient(edges, coords, scores, hole_code, cfg)
    assert np.nanmedian(tf.aux['sampling_conditioning'][tf.valid]) < 0.01


def test_planarity_override_is_used_verbatim():
    from core.estimators import lsq_gradient

    coords, hole_code = _grid()
    scores = np.sin(2 * np.pi * (coords @ TRUE_POLE) / 200.0)
    cfg = _cfg()
    edges = build_edge_set(coords, scores, np.full(len(coords), 2.0),
                           hole_code, cfg)
    tf = lsq_gradient(edges, coords, scores, hole_code, cfg)
    out = field_to_orientations(tf)
    assert out['planarity'].to_numpy() == pytest.approx(tf.aux['gradient_r2'])
