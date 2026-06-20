"""Tests for :mod:`src.tropical_grassmannian` -- four-point violations & scores."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.tropical_grassmannian import (
    compute_violations,
    four_point_sums,
    quadruplet_violation,
    sum_pair_groups,
    tropical_score,
)
from src.utils import generate_quadruplets


def line_metric(points) -> np.ndarray:
    """Additive (tree) metric ``d_ij = |x_i - x_j|`` for points on a line.

    A 1-D embedding is a tree metric, so every quadruplet is four-point
    compatible (violation exactly 0). Used as a hand-built exact example.
    """
    x = np.asarray(points, dtype=float)
    return np.abs(x[:, None] - x[None, :])


def random_symmetric(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 0.0)
    return m


def test_violations_are_non_negative():
    m = random_symmetric(8, seed=0)
    deltas = compute_violations(m, generate_quadruplets(8))
    assert np.all(deltas >= 0.0)


def test_exact_tree_metric_has_zero_violation():
    d = line_metric([0.0, 1.0, 3.0, 7.0, 12.0, 20.0])
    deltas = compute_violations(d, generate_quadruplets(6))
    assert np.max(deltas) < 1e-9

    score = tropical_score(d)
    assert score["max_violation"] < 1e-9
    assert score["l2_loss"] < 1e-12
    assert score["percent_exact"] == pytest.approx(100.0)


def test_single_quadruplet_violation_on_tree_metric():
    d = line_metric([0.0, 2.0, 5.0, 9.0])
    assert quadruplet_violation(d, (0, 1, 2, 3)) < 1e-12


def test_non_tree_metric_has_positive_violation():
    x = np.array([[0, 1, 5, 1], [1, 0, 1, 5], [5, 1, 0, 1], [1, 5, 1, 0]], dtype=float)
    # S1 = 1+1 = 2, S2 = 5+5 = 10, S3 = 1+1 = 2  ->  delta = 10 - 2 = 8
    assert quadruplet_violation(x, (0, 1, 2, 3)) == pytest.approx(8.0)
    assert tropical_score(x)["max_violation"] > 0.0


def test_tropical_score_has_all_keys():
    n = 7
    score = tropical_score(random_symmetric(n, seed=1))
    assert set(score) == {
        "n",
        "n_quadruplets",
        "n_sampled",
        "mean_violation",
        "median_violation",
        "max_violation",
        "l2_loss",
        "percent_exact",
    }
    assert score["n"] == n
    assert score["n_quadruplets"] == math.comb(n, 4)
    assert score["n_sampled"] == math.comb(n, 4)


def test_tropical_score_sampling_is_reproducible():
    m = random_symmetric(20, seed=2)
    s1 = tropical_score(m, sample_size=100, seed=5)
    s2 = tropical_score(m, sample_size=100, seed=5)
    assert s1["n_sampled"] == 100
    assert s1["n_quadruplets"] == math.comb(20, 4)
    assert s1 == s2  # identical results for a fixed seed


def test_tropical_score_small_n_is_vacuously_exact():
    score = tropical_score(np.zeros((3, 3)))
    assert score["n_quadruplets"] == 0
    assert score["n_sampled"] == 0
    assert score["percent_exact"] == 100.0
    assert score["l2_loss"] == 0.0


def test_four_point_sums_and_pair_groups_match_spec():
    x = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]], dtype=float)
    s1, s2, s3 = four_point_sums(x, 0, 1, 2, 3)
    assert s1 == x[0, 1] + x[2, 3]
    assert s2 == x[0, 2] + x[1, 3]
    assert s3 == x[0, 3] + x[1, 2]
    assert sum_pair_groups(0, 1, 2, 3) == (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
