"""Tests for :mod:`src.utils` -- index/vector helpers and numeric utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.utils import (
    generate_quadruplets,
    index_to_pair,
    is_valid_distance_matrix,
    matrix_to_vector,
    num_pairs,
    pair_to_index,
    project_to_distance_space,
    secondmax,
    tropical_norm,
    vector_to_matrix,
)


def test_pair_index_roundtrip_is_bijection():
    for n in range(2, 13):
        p = num_pairs(n)
        seen: set[int] = set()
        for i in range(n):
            for j in range(i + 1, n):
                idx = pair_to_index(i, j, n)
                assert 0 <= idx < p
                assert index_to_pair(idx, n) == (i, j)
                seen.add(idx)
        assert len(seen) == p  # every index hit exactly once


def test_pair_to_index_known_order_n4():
    n = 4
    expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert [pair_to_index(i, j, n) for i, j in expected_pairs] == list(range(6))


def test_pair_to_index_rejects_bad_pair():
    with pytest.raises(ValueError):
        pair_to_index(2, 1, 4)  # i >= j
    with pytest.raises(ValueError):
        pair_to_index(0, 4, 4)  # j out of range


def test_index_to_pair_rejects_out_of_range():
    with pytest.raises(ValueError):
        index_to_pair(6, 4)  # only 6 pairs -> valid indices are 0..5


def test_matrix_vector_roundtrip():
    rng = np.random.default_rng(0)
    for n in range(2, 9):
        m = rng.standard_normal((n, n))
        m = (m + m.T) / 2.0
        np.fill_diagonal(m, 0.0)
        v = matrix_to_vector(m)
        assert v.shape == (num_pairs(n),)
        np.testing.assert_allclose(vector_to_matrix(v, n), m)


def test_matrix_to_vector_ordering():
    m = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]], dtype=float)
    np.testing.assert_allclose(matrix_to_vector(m), [1, 2, 3, 4, 5, 6])


def test_vector_to_matrix_symmetry_and_zero_diagonal():
    v = np.array([1.0, 2.0, 3.0])  # n = 3
    m = vector_to_matrix(v, 3)
    np.testing.assert_allclose(m, m.T)
    np.testing.assert_allclose(np.diag(m), 0.0)


def test_vector_to_matrix_bad_length():
    with pytest.raises(ValueError):
        vector_to_matrix([1.0, 2.0], 3)  # needs length 3


def test_matrix_to_vector_requires_square():
    with pytest.raises(ValueError):
        matrix_to_vector(np.zeros((2, 3)))


def test_secondmax():
    assert secondmax([1, 2, 3]) == 2
    assert secondmax([3, 1, 2]) == 2
    assert secondmax([5, 5, 3]) == 5  # ties with the max return the max
    assert secondmax([7, 7, 7]) == 7
    with pytest.raises(ValueError):
        secondmax([1])


def test_tropical_norm():
    assert tropical_norm([1, 4, 2]) == 3
    assert tropical_norm([0, 0, 0]) == 0
    assert tropical_norm(np.array([-2.0, 3.0])) == 5
    assert tropical_norm([]) == 0.0


def test_generate_quadruplets_full_count():
    for n in range(4, 10):
        q = generate_quadruplets(n, sample_size=None, seed=1)
        assert q.shape == (math.comb(n, 4), 4)
        # rows strictly increasing
        assert np.all(np.diff(q, axis=1) > 0)
        assert np.all((q >= 0) & (q < n))


def test_generate_quadruplets_small_n_is_empty():
    assert generate_quadruplets(3).shape == (0, 4)
    assert generate_quadruplets(0).shape == (0, 4)


def test_generate_quadruplets_sample_size_geq_total_returns_all():
    q = generate_quadruplets(5, sample_size=999, seed=1)  # C(5,4) = 5
    assert q.shape == (5, 4)


def test_generate_quadruplets_sampling_is_seeded_and_unique():
    q1 = generate_quadruplets(20, sample_size=50, seed=42)
    q2 = generate_quadruplets(20, sample_size=50, seed=42)
    q3 = generate_quadruplets(20, sample_size=50, seed=7)

    assert q1.shape == (50, 4)
    np.testing.assert_array_equal(q1, q2)  # reproducible for a fixed seed
    assert not np.array_equal(q1, q3)  # different seed -> different sample
    assert len({tuple(row) for row in q1}) == 50  # all distinct
    assert np.all(np.diff(q1, axis=1) > 0)  # each row strictly increasing
    assert np.all((q1 >= 0) & (q1 < 20))


def test_project_to_distance_space():
    m = np.array([[5.0, -2.0, 1.0], [0.0, 3.0, -4.0], [2.0, 1.0, 9.0]])
    p = project_to_distance_space(m)
    np.testing.assert_allclose(p, p.T)  # symmetric
    np.testing.assert_allclose(np.diag(p), 0.0)  # zero diagonal
    assert np.all(p >= 0.0)  # non-negative
    assert p[0, 1] == 0.0  # (-2 + 0) / 2 = -1 -> clipped to 0
    assert p[0, 2] == 1.5  # (1 + 2) / 2 = 1.5


def test_project_allows_negative_when_flagged():
    m = np.array([[0.0, -1.0], [-1.0, 0.0]])
    p = project_to_distance_space(m, non_negative=False)
    assert p[0, 1] == -1.0


def test_is_valid_distance_matrix():
    good = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    assert is_valid_distance_matrix(good)
    assert not is_valid_distance_matrix(
        np.array([[0.0, 1.0], [2.0, 0.0]])
    )  # asymmetric
    assert not is_valid_distance_matrix(
        np.array([[1.0, 0.0], [0.0, 1.0]])
    )  # nonzero diag
    assert not is_valid_distance_matrix(
        np.array([[0.0, -1.0], [-1.0, 0.0]])
    )  # negative
    assert not is_valid_distance_matrix(np.zeros((2, 3)))  # not square
