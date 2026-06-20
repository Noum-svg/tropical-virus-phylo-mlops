"""Tests for :mod:`src.distances` -- pairwise distance and matrix ``D``."""

from __future__ import annotations

import numpy as np
import pytest

from src.distances import (
    build_distance_matrix,
    distance_matrix_from_clean_df,
    pairwise_distance,
)


def test_identical_sequences_distance_is_zero():
    assert pairwise_distance("ACGTACGT", "ACGTACGT") == 0.0


def test_empty_sequence_edge_cases():
    assert pairwise_distance("", "") == 0.0
    assert pairwise_distance("", "ACGT") == 1.0
    assert pairwise_distance("ACGT", "") == 1.0


def test_distance_is_symmetric():
    a, b = "ACGTACGT", "ACGAACGT"
    assert pairwise_distance(a, b) == pairwise_distance(b, a)


def test_hamming_only_value():
    # equal lengths -> no length penalty; one mismatch out of 4 -> H = 0.25.
    d = pairwise_distance("ACGT", "ACGA", alpha=0.9)
    assert d == pytest.approx(0.9 * 0.25)


def test_length_penalty_only_value():
    # shared prefix matches (H = 0); lengths 4 vs 8 -> P = 4/8 = 0.5.
    d = pairwise_distance("ACGT", "ACGTACGT", alpha=0.9)
    assert d == pytest.approx((1.0 - 0.9) * 0.5)


def test_alpha_bounds_are_validated():
    with pytest.raises(ValueError):
        pairwise_distance("ACGT", "ACGT", alpha=1.5)


def test_distance_values_stay_in_unit_interval():
    rng = np.random.default_rng(0)
    bases = "ACGT"
    seqs = [
        "".join(rng.choice(list(bases), size=int(rng.integers(1, 30))))
        for _ in range(15)
    ]
    matrix = build_distance_matrix(seqs).values
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0


def test_matrix_properties():
    seqs = ["ACGTACGT", "ACGAACGT", "TTTTACGT", "ACGT"]
    names = ["a", "b", "c", "d"]
    frame = build_distance_matrix(seqs, names)
    arr = frame.values

    assert arr.shape == (4, 4)
    np.testing.assert_allclose(np.diag(arr), 0.0)  # zero diagonal
    np.testing.assert_allclose(arr, arr.T)  # symmetric
    assert arr.min() >= 0.0 and arr.max() <= 1.0  # in [0, 1]
    assert list(frame.index) == names
    assert list(frame.columns) == names


def test_default_names():
    frame = build_distance_matrix(["ACGT", "ACGA"])
    assert list(frame.index) == ["seq_0", "seq_1"]


def test_names_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_distance_matrix(["ACGT", "ACGA"], names=["only_one"])


def test_distance_matrix_from_clean_df():
    import pandas as pd

    clean_df = pd.DataFrame(
        {
            "virus_name": ["V1", "V2", "V3"],
            "clean_sequence": ["ACGTACGT", "ACGAACGT", "TTTTTTTT"],
        }
    )
    frame = distance_matrix_from_clean_df(clean_df)
    assert list(frame.index) == ["V1", "V2", "V3"]
    np.testing.assert_allclose(frame.values, frame.values.T)
    np.testing.assert_allclose(np.diag(frame.values), 0.0)
