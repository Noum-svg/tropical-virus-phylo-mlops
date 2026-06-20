"""Small, pure numerical helpers shared across the scientific core.

This module holds index/vector/matrix conversions and tiny numeric utilities
used by :mod:`src.distances`, :mod:`src.tropical_grassmannian`, and
:mod:`src.tropical_gradient_descent`. It depends only on the standard library,
NumPy, and PyYAML, and never imports the delivery layers (``api``/``app``).

Conventions
-----------
The symmetric correction ``omega`` (and any symmetric, zero-diagonal matrix) is
stored as the **upper-triangular vector** of length ``p = n * (n - 1) / 2`` in
row-major order::

    (0, 1), (0, 2), ..., (0, n-1), (1, 2), ..., (n-2, n-1)

:func:`pair_to_index` and :func:`matrix_to_vector` follow exactly this ordering,
and :func:`index_to_pair` / :func:`vector_to_matrix` invert them.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    """Load the central YAML configuration into a nested dictionary.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Path to the YAML configuration file. Defaults to ``"params.yaml"`` in
        the current working directory.

    Returns
    -------
    dict
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Configuration file not found at '{p}'. Expected a params.yaml "
            "with 'data', 'distance', 'four_point', and 'optimization' sections."
        )
    with p.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def num_pairs(n: int) -> int:
    """Return the number of unordered pairs ``p = n * (n - 1) / 2``.

    Parameters
    ----------
    n : int
        Number of items (matrix dimension).

    Returns
    -------
    int
        Length of the upper-triangular vector for an ``n x n`` matrix.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    return n * (n - 1) // 2


def pair_to_index(i: int, j: int, n: int) -> int:
    """Map an unordered pair ``(i, j)`` with ``i < j`` to its vector index.

    Parameters
    ----------
    i, j : int
        Pair coordinates with ``0 <= i < j < n``.
    n : int
        Matrix dimension.

    Returns
    -------
    int
        Index into the upper-triangular vector (see module docstring for the
        ordering).

    Raises
    ------
    ValueError
        If the constraint ``0 <= i < j < n`` is violated.
    """
    if not (0 <= i < j < n):
        raise ValueError(f"Require 0 <= i < j < n, got i={i}, j={j}, n={n}.")
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def index_to_pair(idx: int, n: int) -> tuple[int, int]:
    """Invert :func:`pair_to_index`: map a vector index back to ``(i, j)``.

    Parameters
    ----------
    idx : int
        Index into the upper-triangular vector, ``0 <= idx < n * (n - 1) / 2``.
    n : int
        Matrix dimension.

    Returns
    -------
    tuple of int
        The pair ``(i, j)`` with ``i < j``.

    Raises
    ------
    ValueError
        If ``idx`` is out of range for the given ``n``.
    """
    p = num_pairs(n)
    if not (0 <= idx < p):
        raise ValueError(f"Require 0 <= idx < {p}, got idx={idx} for n={n}.")

    def row_start(r: int) -> int:
        """Vector index of the first entry in row ``r`` (the pair ``(r, r+1)``)."""
        return r * (2 * n - r - 1) // 2

    # Closed-form initial guess via the integer square root, then correct any
    # off-by-one from rounding so the result is always exact.
    i = (2 * n - 1 - math.isqrt((2 * n - 1) ** 2 - 8 * idx)) // 2
    while row_start(i) > idx:
        i -= 1
    while row_start(i + 1) <= idx:
        i += 1
    j = idx - row_start(i) + i + 1
    return int(i), int(j)


def matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """Extract the strict upper triangle of a square matrix as a 1-D vector.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square ``n x n`` matrix. Only the strict upper triangle is read.

    Returns
    -------
    numpy.ndarray
        Vector of length ``n * (n - 1) / 2`` in the ordering described in the
        module docstring.

    Raises
    ------
    ValueError
        If ``matrix`` is not square.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {m.shape}.")
    n = m.shape[0]
    iu = np.triu_indices(n, k=1)
    return m[iu].astype(float).copy()


def vector_to_matrix(vector: Sequence[float] | np.ndarray, n: int) -> np.ndarray:
    """Build a symmetric, zero-diagonal matrix from an upper-triangular vector.

    Parameters
    ----------
    vector : sequence of float or numpy.ndarray
        Upper-triangular entries of length ``n * (n - 1) / 2``.
    n : int
        Matrix dimension.

    Returns
    -------
    numpy.ndarray
        Symmetric ``n x n`` matrix with a zero diagonal.

    Raises
    ------
    ValueError
        If the vector length does not match ``n * (n - 1) / 2``.
    """
    v = np.asarray(vector, dtype=float).ravel()
    expected = num_pairs(n)
    if v.shape[0] != expected:
        raise ValueError(
            f"Vector length {v.shape[0]} does not match n*(n-1)/2 = {expected} "
            f"for n={n}."
        )
    matrix = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n, k=1)
    matrix[iu] = v
    matrix = matrix + matrix.T
    return matrix


def secondmax(values: Sequence[float]) -> float:
    """Return the second-largest value (ties with the maximum return the max).

    For example ``secondmax([5, 5, 3]) == 5`` because the maximum is attained
    twice. This is exactly the quantity subtracted from the maximum to form the
    tropical four-point violation.

    Parameters
    ----------
    values : sequence of float
        At least two values.

    Returns
    -------
    float
        The second-largest value by descending sort.

    Raises
    ------
    ValueError
        If fewer than two values are provided.
    """
    vals = [float(x) for x in values]
    if len(vals) < 2:
        raise ValueError("secondmax requires at least two values.")
    return sorted(vals, reverse=True)[1]


def tropical_norm(g: Sequence[float] | np.ndarray) -> float:
    """Return the tropical norm ``max(g) - min(g)``.

    This is the projective tropical "size" of a vector and is used to scale the
    optimizer step (never the Euclidean norm).

    Parameters
    ----------
    g : sequence of float or numpy.ndarray
        Input vector.

    Returns
    -------
    float
        ``max(g) - min(g)``; ``0.0`` for an empty input.
    """
    arr = np.asarray(g, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    return float(arr.max() - arr.min())


def generate_quadruplets(
    n: int,
    sample_size: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate sorted quadruplets ``(i, j, k, l)`` with ``i < j < k < l``.

    When ``sample_size`` is ``None`` or at least ``C(n, 4)``, every quadruplet
    is returned. Otherwise a reproducible sample of distinct quadruplets is
    drawn without replacement using a seeded NumPy generator. Each draw selects
    four distinct indices uniformly, so every 4-subset is equally likely.

    Parameters
    ----------
    n : int
        Number of items.
    sample_size : int or None, optional
        Desired number of quadruplets. ``None`` means "use all".
    seed : int, optional
        Seed for the NumPy ``default_rng`` used only when sampling.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(num_quadruplets, 4)`` with strictly
        increasing rows. Shape ``(0, 4)`` when ``n < 4``.
    """
    if n < 4:
        return np.empty((0, 4), dtype=int)

    total = math.comb(n, 4)
    if sample_size is None or sample_size >= total:
        return np.array(list(itertools.combinations(range(n), 4)), dtype=int)

    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int, int, int]] = set()
    quads: list[tuple[int, int, int, int]] = []
    while len(quads) < sample_size:
        draw = tuple(int(x) for x in np.sort(rng.choice(n, size=4, replace=False)))
        if draw not in seen:
            seen.add(draw)
            quads.append(draw)
    return np.array(quads, dtype=int)


def project_to_distance_space(
    matrix: np.ndarray, non_negative: bool = True
) -> np.ndarray:
    r"""Project a square matrix onto the distance-matrix space :math:`\mathcal{D}_n`.

    The projection symmetrizes the matrix, optionally clips negative entries to
    zero, and resets the diagonal::

        Pi(M) = ZeroDiag( max(0, (M + M.T) / 2) )

    With ``non_negative=True`` the result satisfies the full invariant
    ``M == M.T``, ``M_ii == 0``, ``M_ij >= 0`` required of every distance-like
    matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square candidate matrix.
    non_negative : bool, optional
        If ``True`` (default) clip negative entries to zero.

    Returns
    -------
    numpy.ndarray
        The projected matrix (a new array).
    """
    m = np.asarray(matrix, dtype=float).copy()
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {m.shape}.")
    m = (m + m.T) / 2.0
    if non_negative:
        m = np.maximum(m, 0.0)
    np.fill_diagonal(m, 0.0)
    return m


def is_valid_distance_matrix(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    """Check the distance-matrix invariants used across module boundaries.

    A valid distance-like matrix is square, finite, symmetric within ``tol``,
    has a zero diagonal within ``tol``, and has no entry below ``-tol``.

    Parameters
    ----------
    matrix : numpy.ndarray
        Candidate matrix.
    tol : float, optional
        Numerical tolerance for the symmetry, diagonal, and non-negativity
        checks. Defaults to ``1e-9``.

    Returns
    -------
    bool
        ``True`` if every invariant holds.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    if not np.all(np.isfinite(m)):
        return False
    if np.any(np.abs(np.diag(m)) > tol):
        return False
    if not np.allclose(m, m.T, atol=tol):
        return False
    if np.any(m < -tol):
        return False
    return True
