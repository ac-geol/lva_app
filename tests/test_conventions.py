"""The eigenvector-assignment guard.

Each estimator declares which eigenvector carries the pole. These tests build
synthetic fields whose correct answer is known by construction and assert the
estimators recover it -- the only defence against a result that is 90 degrees
wrong and still looks plausible on a stereonet.
"""
import numpy as np
import pytest

from core.config import make_cfg
from core.estimators import TensorField, field_to_orientations, shape_pca
from core.geometry import angular_difference, orientation_tensor, unitize
from core.neighbors import EdgeSet, accumulate_tensor, build_edge_set

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

def test_normalized_metrics_are_on_a_common_scale():
    """Two estimators, same perfect structure -> comparable _norm values.

    Both shipped estimators top out at 1.0, so this machinery is dormant today.
    It is what keeps a future estimator whose tensor cannot reach a perfect
    eigenvalue ratio comparable against them, so it stays under test. The 2/3
    and 1/2 ceilings used here are the analytic saturation values of a gradient
    structure tensor under isotropic neighbours -- see docs/METHOD_COMPARISON.md.
    """
    tf, a, _, _ = _diag_field('major', TRUE_POLE, ratio=(1 / 5, 1 / 15, 1 / 15))
    tf.planarity_ceiling = 2.0 / 3.0
    tf.linearity_ceiling = 1.0 / 2.0
    capped = field_to_orientations(tf).iloc[0]

    pca, *_ = _diag_field('minor', TRUE_POLE, ratio=(1.0, 1.0, 0.0))
    sp = field_to_orientations(pca).iloc[0]

    assert capped['planarity'] == pytest.approx(2 / 3, abs=1e-6)   # raw differs
    assert sp['planarity'] == pytest.approx(1.0, abs=1e-6)
    assert capped['planarity_norm'] == pytest.approx(1.0, abs=1e-6)  # norm agrees
    assert sp['planarity_norm'] == pytest.approx(1.0, abs=1e-6)


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


def test_lsq_gradient_beats_naive_tensor_averaging_on_anisotropic_sampling():
    """The whole reason this estimator exists.

    Sampling is restricted to a narrow cone of directions -- the drillhole
    situation. AVERAGING w * delta^2 * u u^T inherits that bias; SOLVING for
    the gradient inverts it away. The averaged tensor is built inline here
    rather than imported: it is the method being beaten, not a shipped
    estimator, and this test is the record of why it is not shipped.
    See docs/METHOD_COMPARISON.md.
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

    def averaged_gradient_tensor():
        """T_i = sum_j w delta^2 u u^T / sum_j w -- average, do not solve."""
        delta = (scores[edges.j] - scores[edges.i]) / edges.d
        wsum = np.bincount(edges.i, weights=edges.w, minlength=edges.n_nodes)
        T = (accumulate_tensor(edges, edges.w * delta ** 2, edges.u)
             / np.where(wsum > 0, wsum, 1.0)[:, None, None])
        valid = edges.neighbor_counts() >= cfg['min_neighbors']
        return TensorField(T=T, pole_eigenvector='major', valid=valid,
                           name='averaged')

    def med_err(tf):
        o = field_to_orientations(tf)
        o = o.loc[o['valid']]
        p = o[['pole_vec_x', 'pole_vec_y', 'pole_vec_z']].to_numpy()
        return float(np.median(
            angular_difference(p, np.broadcast_to(true_pole, p.shape))))

    err_avg = med_err(averaged_gradient_tensor())
    err_ls = med_err(lsq_gradient(edges, coords, scores, hole_code, cfg))

    assert err_ls < 5.0
    assert err_ls < err_avg / 2.0


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
