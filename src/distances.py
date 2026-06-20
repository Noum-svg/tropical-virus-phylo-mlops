"""Pairwise sequence distance and the symmetric distance matrix ``D``.

For two cleaned sequences ``s_i`` and ``s_j`` with lengths ``L_i`` and ``L_j``
and shared comparison length ``L_ij = min(L_i, L_j)``:

.. math::

    H = \\frac{1}{L_{ij}} \\sum_{k=0}^{L_{ij}-1} [\\, s_i[k] \\neq s_j[k]\\,]
    \\qquad
    P = \\frac{|L_i - L_j|}{\\max(L_i, L_j)}
    \\qquad
    d = \\alpha H + (1 - \\alpha) P .

``H`` is the normalized Hamming distance over the shared prefix and ``P`` is a
relative length penalty. The convex combination with ``alpha`` keeps
``d`` in ``[0, 1]``. Edge cases: two empty sequences give ``0.0`` and exactly
one empty sequence gives ``1.0``.

The resulting matrix ``D`` is symmetric with a zero diagonal and entries in
``[0, 1]``; it is returned as a :class:`pandas.DataFrame` indexed by virus name.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_ALPHA: float = 0.9


def pairwise_distance(seq_i: str, seq_j: str, alpha: float = DEFAULT_ALPHA) -> float:
    """Compute the distance between two cleaned sequences.

    Parameters
    ----------
    seq_i, seq_j : str
        Cleaned ``ACGT`` sequences (possibly empty).
    alpha : float, optional
        Weight on the Hamming component; ``1 - alpha`` weights the length
        penalty. Must be in ``[0, 1]``. Defaults to :data:`DEFAULT_ALPHA`.

    Returns
    -------
    float
        Distance in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``alpha`` is outside ``[0, 1]``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

    len_i, len_j = len(seq_i), len(seq_j)
    if len_i == 0 and len_j == 0:
        return 0.0
    if len_i == 0 or len_j == 0:
        return 1.0

    shared = min(len_i, len_j)
    mismatches = sum(1 for k in range(shared) if seq_i[k] != seq_j[k])
    hamming = mismatches / shared
    length_penalty = abs(len_i - len_j) / max(len_i, len_j)
    return float(alpha * hamming + (1.0 - alpha) * length_penalty)


def build_distance_matrix(
    sequences: Sequence[str],
    names: Sequence[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Build the symmetric, zero-diagonal distance matrix ``D``.

    Parameters
    ----------
    sequences : sequence of str
        Cleaned sequences, one per taxon.
    names : sequence of str or None, optional
        Row/column labels. Defaults to ``seq_0, seq_1, ...`` when ``None``.
    alpha : float, optional
        Hamming/length weighting passed to :func:`pairwise_distance`.

    Returns
    -------
    pandas.DataFrame
        ``n x n`` distance matrix indexed and columned by ``names``. It is
        symmetric (explicitly re-symmetrized), has a zero diagonal, and entries
        in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``names`` is provided but its length differs from ``sequences``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    n = len(sequences)
    if names is None:
        names = [f"seq_{i}" for i in range(n)]
    else:
        names = list(names)
        if len(names) != n:
            raise ValueError(
                f"names has length {len(names)} but there are {n} sequences."
            )

    # Encode each sequence once as a uint8 array so the per-pair Hamming count is
    # a vectorized NumPy operation (scales to hundreds/thousands of sequences).
    encoded = [np.frombuffer(s.encode("ascii"), dtype=np.uint8) for s in sequences]
    lengths = [len(s) for s in sequences]

    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        a_i, len_i = encoded[i], lengths[i]
        for j in range(i + 1, n):
            len_j = lengths[j]
            if len_i == 0 and len_j == 0:
                d = 0.0
            elif len_i == 0 or len_j == 0:
                d = 1.0
            else:
                shared = len_i if len_i < len_j else len_j
                mismatches = int(np.count_nonzero(a_i[:shared] != encoded[j][:shared]))
                hamming = mismatches / shared
                length_penalty = abs(len_i - len_j) / max(len_i, len_j)
                d = alpha * hamming + (1.0 - alpha) * length_penalty
            matrix[i, j] = d
            matrix[j, i] = d

    # Enforce the distance-matrix invariants explicitly.
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return pd.DataFrame(matrix, index=names, columns=names)


def distance_matrix_from_clean_df(
    clean_df: pd.DataFrame, alpha: float = DEFAULT_ALPHA
) -> pd.DataFrame:
    """Build ``D`` directly from a cleaned DataFrame.

    Parameters
    ----------
    clean_df : pandas.DataFrame
        Output of :func:`src.data_loader.clean_dataframe`; must contain the
        ``virus_name`` and ``clean_sequence`` columns.
    alpha : float, optional
        Hamming/length weighting passed to :func:`pairwise_distance`.

    Returns
    -------
    pandas.DataFrame
        The distance matrix indexed by ``virus_name``.

    Raises
    ------
    ValueError
        If the required columns are missing.
    """
    required = {"virus_name", "clean_sequence"}
    missing = required - set(clean_df.columns)
    if missing:
        raise ValueError(f"clean_df is missing column(s): {sorted(missing)}.")
    return build_distance_matrix(
        sequences=list(clean_df["clean_sequence"]),
        names=list(clean_df["virus_name"]),
        alpha=alpha,
    )


if __name__ == "__main__":  # pragma: no cover - DVC `distance_matrix` stage
    from pathlib import Path

    from src.utils import load_params

    params = load_params()
    clean_path = Path(params["data"]["processed_dir"]) / "clean_sequences.csv"
    clean_df = pd.read_csv(clean_path)
    frame = distance_matrix_from_clean_df(clean_df, alpha=params["distance"]["alpha"])
    out = Path(params["outputs"]["output_dir"]) / "distance_matrix.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out)
    print(f"[distances] wrote {frame.shape[0]}x{frame.shape[1]} matrix to {out}")
