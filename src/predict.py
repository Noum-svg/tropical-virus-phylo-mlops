"""End-to-end orchestration: sequences -> D -> X -> phylogenetic tree.

This module only *composes* the scientific core (:mod:`src.data_loader`,
:mod:`src.distances`, :mod:`src.tropical_gradient_descent`,
:mod:`src.phylogeny`). It contains no distance, four-point, optimization, or
Neighbor-Joining math of its own.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_loader import clean_dataframe
from src.distances import distance_matrix_from_clean_df
from src.phylogeny import reconstruct_tree
from src.tropical_gradient_descent import correct_distance_matrix


def predict_from_sequences(
    virus_names: list[str],
    sequences: list[str],
    *,
    min_seq_length: int = 1,
    alpha: float = 0.9,
    epochs: int = 200,
    gamma: float = 0.05,
    lambda_reg: float = 0.01,
    epsilon: float = 1e-8,
    quadruplet_sample_size: int | None = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full pipeline on in-memory sequences.

    Parameters
    ----------
    virus_names : list of str
        Taxon names, aligned with ``sequences``.
    sequences : list of str
        Raw RNA/DNA sequences (cleaned internally; never fabricated).
    min_seq_length : int, optional
        Drop cleaned sequences shorter than this. Defaults to ``1``.
    alpha : float, optional
        Hamming/length weighting for the distance. Defaults to ``0.9``.
    epochs, gamma, lambda_reg, epsilon, quadruplet_sample_size, seed
        Tropical Gradient Descent configuration.

    Returns
    -------
    dict
        Keys: ``virus_names``, ``clean_sequences``, ``sequence_lengths``,
        ``distance_matrix`` (DataFrame), ``corrected_distance_matrix``
        (DataFrame), ``omega`` (DataFrame), ``metrics_before``,
        ``metrics_after``, ``relative_improvement``, ``tree`` (:class:`PhyloTree`),
        ``tree_newick``, ``tree_edges`` (list of dict), ``tree_dot``, and
        ``history`` (DataFrame). NumPy/pandas objects are returned as-is; the API
        layer serializes them at the JSON boundary.

    Raises
    ------
    ValueError
        If the input lengths mismatch or no sequence survives cleaning.
    """
    if len(virus_names) != len(sequences):
        raise ValueError(
            f"virus_names ({len(virus_names)}) and sequences ({len(sequences)}) "
            "must have the same length."
        )

    raw = pd.DataFrame({"virus_name": virus_names, "rna_sequence": sequences})
    clean_df = clean_dataframe(raw, min_seq_length=min_seq_length)
    if len(clean_df) == 0:
        raise ValueError(
            "No sequence survived cleaning/filtering "
            f"(min_seq_length={min_seq_length}). Provide valid ACGT/U sequences."
        )

    names = list(clean_df["virus_name"])
    distance_matrix = distance_matrix_from_clean_df(clean_df, alpha=alpha)

    config = {
        "epochs": epochs,
        "gamma": gamma,
        "lambda_reg": lambda_reg,
        "epsilon": epsilon,
        "quadruplet_sample_size": quadruplet_sample_size,
        "seed": seed,
    }
    correction = correct_distance_matrix(distance_matrix.to_numpy(), config)

    corrected = pd.DataFrame(correction["X"], index=names, columns=names)
    omega = pd.DataFrame(correction["omega"], index=names, columns=names)
    tree = reconstruct_tree(corrected)

    return {
        "virus_names": names,
        "clean_sequences": list(clean_df["clean_sequence"]),
        "sequence_lengths": [int(x) for x in clean_df["sequence_length"]],
        "distance_matrix": distance_matrix,
        "corrected_distance_matrix": corrected,
        "omega": omega,
        "metrics_before": correction["metrics_before"],
        "metrics_after": correction["metrics_after"],
        "relative_improvement": correction["relative_improvement"],
        "tree": tree,
        "tree_newick": tree.to_newick(),
        "tree_edges": tree.to_edge_list(),
        "tree_dot": tree.to_dot(),
        "history": correction["history"],
    }
