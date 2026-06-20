r"""Tropical four-point sums, violations, and global compatibility metrics.

For a symmetric matrix ``X`` and a quadruplet ``i < j < k < l`` the three
"split sums" are::

    S1 = X[i, j] + X[k, l]      (pair group 0)
    S2 = X[i, k] + X[j, l]      (pair group 1)
    S3 = X[i, l] + X[j, k]      (pair group 2)

The tropical four-point violation is

.. math::

    \delta_{ijkl}(X) = \max(S_1, S_2, S_3) - \operatorname{secondmax}(S_1, S_2, S_3)
    \;\ge\; 0 .

``X`` is tree-compatible on the quadruplet exactly when ``delta == 0`` (the
maximum is attained at least twice); numerically this is tested as
``delta <= eps``. This is the tropical (Speyer--Sturmfels) characterization of
tree metrics via the Grassmannian ``Gr(2, n)``.

This module computes violations and aggregate scores only. The optimization
subgradient that consumes :func:`sum_pair_groups` lives in
:mod:`src.tropical_gradient_descent`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.utils import generate_quadruplets

#: Default numerical tolerance for treating a quadruplet as exactly compatible.
DEFAULT_EPS: float = 1e-9


def sum_pair_groups(
    i: int, j: int, k: int, l: int
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Return the three unordered pair groups whose entry sums form S1, S2, S3.

    Parameters
    ----------
    i, j, k, l : int
        Quadruplet indices, expected sorted as ``i < j < k < l``.

    Returns
    -------
    tuple
        ``(((i, j), (k, l)), ((i, k), (j, l)), ((i, l), (j, k)))`` -- the pairs
        forming ``S1``, ``S2``, ``S3`` respectively.
    """
    return (((i, j), (k, l)), ((i, k), (j, l)), ((i, l), (j, k)))


def four_point_sums(
    x: np.ndarray, i: int, j: int, k: int, l: int
) -> tuple[float, float, float]:
    """Compute the three split sums ``(S1, S2, S3)`` for one quadruplet.

    Parameters
    ----------
    x : numpy.ndarray
        Symmetric matrix.
    i, j, k, l : int
        Quadruplet indices.

    Returns
    -------
    tuple of float
        ``(S1, S2, S3)``.
    """
    s1 = float(x[i, j] + x[k, l])
    s2 = float(x[i, k] + x[j, l])
    s3 = float(x[i, l] + x[j, k])
    return s1, s2, s3


def quadruplet_violation(x: np.ndarray, quad: tuple[int, int, int, int]) -> float:
    """Return ``delta`` for a single quadruplet ``(i, j, k, l)``.

    Parameters
    ----------
    x : numpy.ndarray
        Symmetric matrix.
    quad : tuple of int
        The four indices.

    Returns
    -------
    float
        ``max - secondmax`` of the three split sums; always ``>= 0``.
    """
    i, j, k, l = quad
    sums = sorted(four_point_sums(x, i, j, k, l))  # ascending
    return float(sums[2] - sums[1])  # max - secondmax


def compute_violations(x: np.ndarray, quadruplets: np.ndarray) -> np.ndarray:
    """Vectorized ``delta`` over an array of quadruplets.

    Parameters
    ----------
    x : numpy.ndarray
        Symmetric ``n x n`` matrix.
    quadruplets : numpy.ndarray
        Integer array of shape ``(m, 4)`` with rows ``(i, j, k, l)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(m,)`` of non-negative violations. Empty input yields
        an empty array.
    """
    x = np.asarray(x, dtype=float)
    quads = np.asarray(quadruplets, dtype=int)
    if quads.size == 0:
        return np.empty((0,), dtype=float)

    i, j, k, l = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    s1 = x[i, j] + x[k, l]
    s2 = x[i, k] + x[j, l]
    s3 = x[i, l] + x[j, k]
    sums = np.stack([s1, s2, s3], axis=1)
    sums.sort(axis=1)  # ascending along each row
    return sums[:, 2] - sums[:, 1]  # max - secondmax >= 0


def tropical_score(
    x: np.ndarray,
    eps: float = DEFAULT_EPS,
    sample_size: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate tropical four-point metrics for a matrix.

    When ``C(n, 4) > sample_size`` the quadruplets are sampled reproducibly
    without replacement using ``seed``; otherwise all quadruplets are used.

    Parameters
    ----------
    x : numpy.ndarray
        Symmetric ``n x n`` matrix.
    eps : float, optional
        Tolerance: a quadruplet with ``delta < eps`` counts as exact.
    sample_size : int or None, optional
        Number of quadruplets to evaluate; ``None`` uses all.
    seed : int, optional
        Seed for reproducible quadruplet sampling.

    Returns
    -------
    dict
        Keys: ``n``, ``n_quadruplets`` (total ``C(n, 4)``), ``n_sampled``
        (evaluated count), ``mean_violation``, ``median_violation``,
        ``max_violation``, ``l2_loss`` (``sum(delta**2)``), and
        ``percent_exact`` (``100 * #{delta < eps} / n_sampled``).

    Notes
    -----
    For ``n < 4`` there are no quadruplets; all violation statistics are ``0``
    and ``percent_exact`` is ``100.0`` (vacuously compatible).
    """
    x = np.asarray(x, dtype=float)
    n = int(x.shape[0])
    n_quadruplets = math.comb(n, 4) if n >= 4 else 0
    quads = generate_quadruplets(n, sample_size=sample_size, seed=seed)
    n_sampled = int(quads.shape[0])

    if n_sampled == 0:
        return {
            "n": n,
            "n_quadruplets": int(n_quadruplets),
            "n_sampled": 0,
            "mean_violation": 0.0,
            "median_violation": 0.0,
            "max_violation": 0.0,
            "l2_loss": 0.0,
            "percent_exact": 100.0,
        }

    deltas = compute_violations(x, quads)
    return {
        "n": n,
        "n_quadruplets": int(n_quadruplets),
        "n_sampled": n_sampled,
        "mean_violation": float(deltas.mean()),
        "median_violation": float(np.median(deltas)),
        "max_violation": float(deltas.max()),
        "l2_loss": float(np.sum(deltas**2)),
        "percent_exact": float(100.0 * np.count_nonzero(deltas < eps) / n_sampled),
    }
