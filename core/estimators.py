"""The two orientation estimators, and the one place their conventions differ.

  ############################################################################
  #  READ THIS BEFORE TOUCHING ANYTHING BELOW                                #
  #                                                                          #
  #  The eigenvector that carries the POLE IS NOT THE SAME between these     #
  #  estimators:                                                             #
  #                                                                          #
  #    shape_pca        -- tensor is a covariance of POSITIONS.              #
  #                        major eigenvector = long axis   = LINEATION       #
  #                        minor eigenvector = short axis  = POLE            #
  #                                                                          #
  #    lsq_gradient     -- tensor is the outer product of a SOLVED gradient. #
  #                        major eigenvector = gradient direction = POLE     #
  #                        minor eigenvector = LINEATION                     #
  #                                                                          #
  #  Getting this backwards produces a result that is silently 90 degrees    #
  #  wrong and looks entirely plausible on a stereonet. Each estimator       #
  #  therefore *declares* its convention as TensorField.pole_eigenvector and #
  #  the conversion to angles happens in exactly one function, which trusts  #
  #  that declaration. tests/test_conventions.py locks all of it down with   #
  #  synthetic geometry of known answer.                                     #
  #                                                                          #
  #  SECOND TRAP: confidence metrics need not share a scale. An estimator    #
  #  whose tensor cannot reach a perfect ratio declares its ceilings, and    #
  #  the normalized planarity_norm / linearity_norm columns are what a       #
  #  method comparison compares. Both estimators here top out at 1.0, so     #
  #  the machinery is currently dormant -- it is retained because it is the  #
  #  only thing that keeps a future estimator honest against these two.      #
  #  See CLAUDE.md section 3 for the estimators this replaced.               #
  ############################################################################
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .geometry import normal_to_strike_dip, vector_to_trend_plunge
from .neighbors import EdgeSet, accumulate_tensor




@dataclass
class TensorField:
    """A per-node 3x3 symmetric tensor plus the convention needed to read it."""
    T: np.ndarray                       # (n_nodes, 3, 3)
    pole_eigenvector: str               # 'major' or 'minor'
    valid: np.ndarray                   # bool mask, (n_nodes,)
    name: str
    # Value the planarity/linearity metrics reach for a *perfect* structure of
    # that kind. Used to put every estimator on a common 0-1 scale.
    planarity_ceiling: float = 1.0
    linearity_ceiling: float = 1.0
    # Estimators whose confidence is not an eigenvalue ratio supply it here.
    planarity_override: np.ndarray | None = None
    aux: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.pole_eigenvector not in ('major', 'minor'):
            raise ValueError(
                f"pole_eigenvector must be 'major' or 'minor', "
                f"got {self.pole_eigenvector!r}")


def _validity(edges: EdgeSet, hole_code: np.ndarray, cfg: dict) -> np.ndarray:
    """A node is estimable only with enough neighbours from enough holes."""
    ok = edges.neighbor_counts() >= cfg['min_neighbors']
    min_holes = cfg.get('min_holes', 0)
    if min_holes:
        ok &= edges.hole_counts(hole_code) >= min_holes
    return ok


def shape_pca(edges: EdgeSet, coords: np.ndarray, scores: np.ndarray,
              hole_code: np.ndarray, cfg: dict) -> TensorField:
    """Weighted covariance of neighbour POSITIONS -- the notebook's estimator.

    Measures the shape of the sampled point cloud. Kept as the reference
    method: see CLAUDE.md section 3 for why it
    reports the drill pattern rather than the geology.
    """
    n = edges.n_nodes
    W = np.bincount(edges.i, weights=edges.w, minlength=n)
    Wsafe = np.where(W > 0, W, 1.0)

    cj = coords[edges.j]
    mean = np.column_stack([
        np.bincount(edges.i, weights=edges.w * cj[:, a], minlength=n) / Wsafe
        for a in range(3)])

    M = accumulate_tensor(edges, edges.w, cj) / Wsafe[:, None, None]
    T = M - np.einsum('na,nb->nab', mean, mean)

    return TensorField(T=T, pole_eigenvector='minor',
                       valid=_validity(edges, hole_code, cfg),
                       name='shape_pca')


def lsq_gradient(edges: EdgeSet, coords: np.ndarray, scores: np.ndarray,
                 hole_code: np.ndarray, cfg: dict) -> TensorField:
    """Least-squares gradient reconstruction -- solve, do not average.

    The structure tensor forms  <w * delta^2 * u u^T>. That average is only an
    unbiased estimate of the gradient direction when the neighbour directions u
    are ISOTROPIC. Drillhole sampling is the opposite of isotropic: which
    directions exist at all is decided by the drill programme, so the direction
    of largest observed contrast is partly a statement about where the other
    holes are. On the test property that costs ~50 degrees even on a noise-free
    lens (see CLAUDE.md section 3).

    Solving instead of averaging removes it. Minimizing

        sum_j w_ij (delta_ij - grad . u_ij)^2

    gives the normal equations

        (sum_j w_ij u_ij u_ij^T) grad = sum_j w_ij delta_ij u_ij

    where the matrix on the left is precisely the sampling-direction tensor.
    Inverting it deconvolves the drill geometry rather than averaging over it.

    The pole is the gradient direction. Confidence is `gradient_r2`, the share
    of observed grade change a single linear gradient explains -- an honest
    measure, unlike an eigenvalue ratio of a rank-one tensor, which is always 1.
    `sampling_conditioning` (lambda_min/lambda_max of the direction tensor)
    reports when the drill pattern cannot constrain a gradient at all.
    """
    n = edges.n_nodes
    delta = (scores[edges.j] - scores[edges.i]) / edges.d

    A = np.zeros((n, 3, 3))
    b = np.zeros((n, 3))
    for a in range(3):
        b[:, a] = np.bincount(edges.i, weights=edges.w * delta * edges.u[:, a],
                              minlength=n)
        for c in range(a, 3):
            comp = np.bincount(edges.i,
                               weights=edges.w * edges.u[:, a] * edges.u[:, c],
                               minlength=n)
            A[:, a, c] = comp
            A[:, c, a] = comp

    W = np.bincount(edges.i, weights=edges.w, minlength=n)
    Wsafe = np.where(W > 0, W, 1.0)
    A /= Wsafe[:, None, None]
    b /= Wsafe[:, None]

    ev = np.linalg.eigvalsh(A)                       # ascending
    conditioning = np.divide(ev[:, 0], ev[:, 2],
                             out=np.zeros(n), where=ev[:, 2] > 0)

    # Ridge scaled to each node's own magnitude, so it regularizes shape rather
    # than size and never dominates a well-conditioned node.
    ridge = cfg.get('lsq_ridge', 1e-3)
    scale = np.trace(A, axis1=1, axis2=2) / 3.0
    reg = A + (ridge * np.where(scale > 0, scale, 1.0))[:, None, None] * np.eye(3)
    grad = np.linalg.solve(reg, b[:, :, None])[:, :, 0]

    # How much of the observed contrast one linear gradient accounts for.
    pred = np.einsum('ea,ea->e', grad[edges.i], edges.u)
    ss_res = np.bincount(edges.i, weights=edges.w * (delta - pred) ** 2, minlength=n)
    ss_tot = np.bincount(edges.i, weights=edges.w * delta ** 2, minlength=n)
    r2 = np.clip(1.0 - np.divide(ss_res, ss_tot,
                                 out=np.zeros(n), where=ss_tot > 0), 0.0, 1.0)

    gmag = np.linalg.norm(grad, axis=1)
    T = np.einsum('na,nb->nab', grad, grad)

    valid = _validity(edges, hole_code, cfg)
    valid &= gmag > 0
    min_cond = cfg.get('min_sampling_conditioning', 0.0)
    if min_cond:
        valid &= conditioning >= min_cond

    return TensorField(T=T, pole_eigenvector='major', valid=valid,
                       name='lsq_gradient', planarity_override=r2,
                       aux={'gradient_r2': r2,
                            'sampling_conditioning': conditioning,
                            'gradient_magnitude': gmag})


ESTIMATORS = {
    'shape_pca': shape_pca,
    'lsq_gradient': lsq_gradient,
}


def field_to_orientations(tf: TensorField) -> pd.DataFrame:
    """Eigen-decompose a tensor field and convert to geological angles.

    The single place the pole/lineation convention is applied. Everything
    downstream reads plain column names and needs to know nothing about which
    estimator produced them.

    planarity / linearity are defined by MEANING, not by eigenvalue position,
    so they stay comparable across estimators:
      planarity -- confidence that a single plane is defined here
      linearity -- confidence that a single line is defined here

    planarity_norm / linearity_norm divide those by the estimator's declared
    ceiling, so 1.0 means "perfect structure of this kind" for every estimator.
    Compare estimators on the _norm columns; interpret one estimator's field on
    either.
    """
    evals, evecs = np.linalg.eigh(tf.T)          # ascending
    evals = evals[:, ::-1]                       # -> descending
    evecs = evecs[:, :, ::-1]
    l1, l2, l3 = evals[:, 0], evals[:, 1], evals[:, 2]

    major, minor = evecs[:, :, 0], evecs[:, :, 2]

    if tf.pole_eigenvector == 'major':
        pole, line = major, minor
        # One dominant tensor axis => one well-defined pole => a real plane.
        planarity = np.divide(l1 - l2, l1, out=np.full_like(l1, np.nan), where=l1 > 0)
        linearity = np.divide(l2 - l3, l1, out=np.full_like(l1, np.nan), where=l1 > 0)
    else:
        pole, line = minor, major
        planarity = np.divide(l2 - l3, l1, out=np.full_like(l1, np.nan), where=l1 > 0)
        linearity = np.divide(l1 - l2, l1, out=np.full_like(l1, np.nan), where=l1 > 0)

    line_trend, line_plunge = vector_to_trend_plunge(line)
    pole_trend, pole_plunge = vector_to_trend_plunge(pole)
    strike, dip = normal_to_strike_dip(pole)

    if tf.planarity_override is not None:
        planarity = np.asarray(tf.planarity_override, dtype=float)

    # Rescale so 1.0 means "as good as this estimator can possibly report".
    planarity_norm = np.clip(planarity / tf.planarity_ceiling, 0.0, 1.0)
    linearity_norm = np.clip(linearity / tf.linearity_ceiling, 0.0, 1.0)

    out = pd.DataFrame({
        'estimator': tf.name,
        'eig1': l1, 'eig2': l2, 'eig3': l3,
        'planarity': planarity, 'linearity': linearity,
        'planarity_norm': planarity_norm, 'linearity_norm': linearity_norm,
        'line_vec_x': line[:, 0], 'line_vec_y': line[:, 1], 'line_vec_z': line[:, 2],
        'pole_vec_x': pole[:, 0], 'pole_vec_y': pole[:, 1], 'pole_vec_z': pole[:, 2],
        'line_trend_deg': line_trend, 'line_plunge_deg': line_plunge,
        'pole_trend_deg': pole_trend, 'pole_plunge_deg': pole_plunge,
        'plane_strike_deg': strike, 'plane_dip_deg': dip,
        'valid': tf.valid,
    })
    for k, v in tf.aux.items():
        if isinstance(v, np.ndarray) and len(v) == len(out):
            out[k] = v
    return out


def apply_smoothing(tf: TensorField, edges, *, mu: float = 0.0,
                    iterations: int = 10) -> TensorField:
    """Laplacian-smooth a tensor field, preserving its declared conventions."""
    from .smoothing import smooth_tensor_field
    if mu <= 0:
        return tf
    return TensorField(
        T=smooth_tensor_field(tf.T, edges, mu=mu, iterations=iterations),
        pole_eigenvector=tf.pole_eigenvector, valid=tf.valid, name=tf.name,
        planarity_ceiling=tf.planarity_ceiling,
        linearity_ceiling=tf.linearity_ceiling,
        planarity_override=tf.planarity_override, aux=tf.aux)
