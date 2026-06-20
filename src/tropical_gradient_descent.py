r"""Tropical Gradient Descent: learn the correction ``omega`` so that ``X = D + omega``.

The correction is stored as the upper-triangular vector ``w`` of length
``p = n (n - 1) / 2`` (see :mod:`src.utils`). The corrected matrix is
``X = D + vector_to_matrix(w)``.

Objective (sum form, with a Frobenius-type penalty on the correction vector)::

    L(w) = sum_q  delta_q(D + omega)^2  +  lambda_reg * ||w||_2^2

where ``delta_q`` is the tropical four-point violation. Since ``w`` holds each
off-diagonal pair once, ``||w||_2^2 = (1/2) ||omega||_F^2``; the constant is
absorbed into ``lambda_reg``.

Practical subgradient (NumPy only -- no PyTorch/TensorFlow). For each quadruplet
with ``delta > eps``, let ``coeff = 2 * delta``; add ``+coeff`` to the two
vector entries of the maximal-sum pair group and ``-coeff`` to the two entries
of the second-maximal group (signs are opposite). Ties on the maximum give
``delta ~ 0`` and contribute nothing; the deterministic ordering breaks ties by
lowest group index. Finally add the regularization gradient ``2 * lambda_reg * w``.

Optimizer (the **Hybrid** variant chosen for this project):

* a *normalized* subgradient step that uses the **tropical** norm of the
  gradient (never the Euclidean norm), ``eta = gamma / (tropical_norm(g) + epsilon)``,
  so the update is ``w <- w - eta * g``;
* stop when the gradient's tropical norm drops to ``<= epsilon``;
* after each step, project ``X = D + omega`` back onto the distance space
  (symmetrize, clip negatives to keep ``X >= 0``, zero the diagonal) and recover
  ``omega = X - D``. This preserves the matrix invariants every epoch;
* return the **best iterate** seen (lowest loss, which always includes the
  initial ``omega = 0``), so the corrected matrix is never worse than ``D``.

Why normalize *by* the tropical norm? The data-gradient is zero-sum (each
quadruplet contributes ``+2d, +2d, -2d, -2d``), so its tropical norm
``max(g) - min(g)`` is a faithful magnitude. Dividing by it yields a bounded,
scale-invariant step of size ~``gamma``. (The literally written
``eta = gamma * tropical_norm(g) / sqrt(p)`` puts the norm in the numerator and
diverges -- the step then grows with the gradient -- which contradicts the
spec's own "loss must not increase" requirement; this implementation uses the
stable normalized form instead.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.tropical_grassmannian import tropical_score
from src.utils import (
    generate_quadruplets,
    matrix_to_vector,
    num_pairs,
    project_to_distance_space,
    tropical_norm,
    vector_to_matrix,
)

HISTORY_COLUMNS: tuple[str, ...] = ("epoch", "loss", "grad_tr_norm", "eta")


def _resolve_quadruplets(config: dict[str, Any], n: int) -> np.ndarray:
    """Return the fixed quadruplet set for a run (precomputed or generated)."""
    quads = config.get("quadruplets")
    if quads is None:
        quads = generate_quadruplets(
            n,
            sample_size=config.get("quadruplet_sample_size"),
            seed=int(config.get("seed", 42)),
        )
    return np.asarray(quads, dtype=int)


def four_point_loss_and_subgradient(
    D: np.ndarray, omega: np.ndarray, config: dict[str, Any]
) -> tuple[float, np.ndarray]:
    """Compute the loss and a practical subgradient at the correction ``omega``.

    Parameters
    ----------
    D : numpy.ndarray
        Base distance matrix (``n x n``).
    omega : numpy.ndarray
        Current correction as an upper-triangular vector of length ``p``.
    config : dict
        Reads ``lambda_reg``, ``epsilon`` (tolerance), and either a precomputed
        ``quadruplets`` array or ``quadruplet_sample_size`` + ``seed``.

    Returns
    -------
    loss : float
        ``sum(delta**2) + lambda_reg * ||omega||_2^2``.
    grad : numpy.ndarray
        Subgradient with the same shape as ``omega`` (length ``p``).
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    w = np.asarray(omega, dtype=float).ravel()
    p = num_pairs(n)
    lam = float(config.get("lambda_reg", 0.001))
    eps = float(config.get("epsilon", 1e-9))
    quads = _resolve_quadruplets(config, n)

    grad = np.zeros(p, dtype=float)
    if quads.size == 0:  # n < 4 (or empty sample): only the regularizer acts
        grad += 2.0 * lam * w
        return float(lam * np.dot(w, w)), grad

    x = D + vector_to_matrix(w, n)
    i, j, k, l = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]

    # Three split sums; pair groups: 0->(ij,kl), 1->(ik,jl), 2->(il,jk).
    s = np.stack([x[i, j] + x[k, l], x[i, k] + x[j, l], x[i, l] + x[j, k]], axis=1)
    order = np.argsort(-s, axis=1, kind="stable")  # descending; ties -> lowest group
    rows = np.arange(s.shape[0])
    g_max, g_sec = order[:, 0], order[:, 1]
    deltas = s[rows, g_max] - s[rows, g_sec]  # max - secondmax >= 0

    loss = float(np.sum(deltas**2) + lam * np.dot(w, w))

    def pidx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorized pair_to_index for a < b (upper-triangular ordering)."""
        return a * (2 * n - a - 1) // 2 + (b - a - 1)

    # Vector indices of the two pairs in each group, shape (m, 3, 2).
    pairs = np.stack(
        [
            np.stack([pidx(i, j), pidx(k, l)], axis=1),
            np.stack([pidx(i, k), pidx(j, l)], axis=1),
            np.stack([pidx(i, l), pidx(j, k)], axis=1),
        ],
        axis=1,
    )
    max_pairs = pairs[rows, g_max]  # (m, 2)
    sec_pairs = pairs[rows, g_sec]  # (m, 2)
    coeff = 2.0 * deltas
    mask = deltas > eps

    # +coeff on the max group's two entries, -coeff on the second group's.
    np.add.at(grad, max_pairs[mask, 0], coeff[mask])
    np.add.at(grad, max_pairs[mask, 1], coeff[mask])
    np.add.at(grad, sec_pairs[mask, 0], -coeff[mask])
    np.add.at(grad, sec_pairs[mask, 1], -coeff[mask])
    grad += 2.0 * lam * w  # regularization gradient
    return loss, grad


def fit_tropical_gradient_descent(
    D: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Run Hybrid Tropical Gradient Descent.

    Parameters
    ----------
    D : numpy.ndarray
        Base distance matrix (``n x n``).
    config : dict
        Reads ``epochs``, ``gamma``, ``lambda_reg``, ``epsilon``,
        ``quadruplet_sample_size``, ``seed``.

    Returns
    -------
    omega : numpy.ndarray
        Learned correction as an upper-triangular vector of length ``p``.
    X : numpy.ndarray
        Corrected, projected distance matrix ``X = Pi(D + omega)``.
    history : pandas.DataFrame
        One row per epoch with columns :data:`HISTORY_COLUMNS`.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    p = num_pairs(n)

    epochs = int(config.get("epochs", 1000))
    gamma = float(config.get("gamma", 0.05))
    epsilon = float(config.get("epsilon", 1e-9))

    # Freeze one quadruplet sample for the whole run so losses stay comparable.
    run_cfg = dict(config)
    run_cfg["quadruplets"] = _resolve_quadruplets(config, n)

    w = np.zeros(p, dtype=float)
    best_w = w.copy()
    best_loss = np.inf
    history: list[dict[str, float]] = []

    if p == 0:  # n < 2: nothing to optimize
        return w, project_to_distance_space(D), pd.DataFrame(columns=HISTORY_COLUMNS)

    for t in range(1, epochs + 1):
        loss, grad = four_point_loss_and_subgradient(D, w, run_cfg)
        if loss < best_loss:  # best-iterate tracking (subgradient is non-monotone)
            best_loss = loss
            best_w = w.copy()
        tr = tropical_norm(grad)
        if tr <= epsilon:  # converged
            history.append({"epoch": t, "loss": loss, "grad_tr_norm": tr, "eta": 0.0})
            break
        eta = gamma / (tr + epsilon)  # normalized tropical-norm step
        history.append({"epoch": t, "loss": loss, "grad_tr_norm": tr, "eta": eta})
        w = w - eta * grad
        # Hybrid projection: keep X symmetric, non-negative, zero-diagonal.
        x = project_to_distance_space(D + vector_to_matrix(w, n), non_negative=True)
        w = matrix_to_vector(x - D)

    # Return the best iterate seen (includes w = 0), so X is never worse than D.
    x_best = project_to_distance_space(
        D + vector_to_matrix(best_w, n), non_negative=True
    )
    return best_w, x_best, pd.DataFrame(history, columns=list(HISTORY_COLUMNS))


def correct_distance_matrix(D: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Fit the correction and report before/after metrics and improvement.

    Parameters
    ----------
    D : numpy.ndarray
        Base distance matrix.
    config : dict
        Optimization configuration (see :func:`fit_tropical_gradient_descent`).

    Returns
    -------
    dict
        ``omega`` (matrix), ``omega_vector``, ``X`` (corrected matrix),
        ``metrics_before``, ``metrics_after``, ``relative_improvement``, and
        ``history``. The before/after scores use the same quadruplet sample for
        comparability.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    eps = float(config.get("epsilon", 1e-9))
    sample_size = config.get("quadruplet_sample_size")
    seed = int(config.get("seed", 42))

    omega_vec, x, history = fit_tropical_gradient_descent(D, config)

    before = tropical_score(D, eps=eps, sample_size=sample_size, seed=seed)
    after = tropical_score(x, eps=eps, sample_size=sample_size, seed=seed)
    relative_improvement = (before["l2_loss"] - after["l2_loss"]) / (
        before["l2_loss"] + eps
    )

    return {
        "omega": vector_to_matrix(omega_vec, n) if n > 1 else np.zeros((n, n)),
        "omega_vector": omega_vec,
        "X": x,
        "metrics_before": before,
        "metrics_after": after,
        "relative_improvement": float(relative_improvement),
        "history": history,
    }
