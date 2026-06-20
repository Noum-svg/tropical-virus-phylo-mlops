"""Tests for :mod:`src.tropical_gradient_descent` -- loss, subgradient, fit."""

from __future__ import annotations

import numpy as np
import pytest

from src.tropical_gradient_descent import (
    HISTORY_COLUMNS,
    correct_distance_matrix,
    fit_tropical_gradient_descent,
    four_point_loss_and_subgradient,
)
from src.tropical_grassmannian import tropical_score
from src.utils import is_valid_distance_matrix, num_pairs, project_to_distance_space


def line_metric(points) -> np.ndarray:
    """Additive (tree) metric ``d_ij = |x_i - x_j|`` -- exact, zero violations."""
    x = np.asarray(points, dtype=float)
    return np.abs(x[:, None] - x[None, :])


def noisy_tree_metric(seed: int = 0) -> np.ndarray:
    """A valid distance matrix that perturbs an exact tree metric."""
    rng = np.random.default_rng(seed)
    base = line_metric([0, 1, 3, 6, 10, 15, 21])
    noise = rng.normal(0, 0.4, base.shape)
    noise = (noise + noise.T) / 2.0
    np.fill_diagonal(noise, 0.0)
    return project_to_distance_space(base + noise)


CONFIG = dict(
    epochs=600,
    gamma=0.05,
    lambda_reg=0.001,
    epsilon=1e-9,
    quadruplet_sample_size=None,
    seed=42,
)


def test_subgradient_shape_matches_omega():
    d = noisy_tree_metric()
    p = num_pairs(d.shape[0])
    w = np.zeros(p)
    loss, grad = four_point_loss_and_subgradient(
        d, w, {"lambda_reg": 0.001, "epsilon": 1e-9}
    )
    assert grad.shape == w.shape == (p,)
    assert isinstance(loss, float)


def test_subgradient_signs_and_values():
    # Single quadruplet with a unique max group (S2). Expect +2*delta on the
    # max group's two pair entries and -2*delta on the second group's.
    d = np.array([[0, 1, 5, 1], [1, 0, 1, 5], [5, 1, 0, 1], [1, 5, 1, 0]], dtype=float)
    w = np.zeros(6)
    loss, grad = four_point_loss_and_subgradient(
        d, w, {"lambda_reg": 0.001, "epsilon": 1e-9}
    )
    # S1=2, S2=10, S3=2 -> delta=8, coeff=16. Pair order for n=4:
    # 0:(0,1) 1:(0,2) 2:(0,3) 3:(1,2) 4:(1,3) 5:(2,3)
    # max group S2 -> (0,2),(1,3) = idx 1,4 (+16); second S1 -> (0,1),(2,3) = idx 0,5 (-16)
    np.testing.assert_allclose(grad, [-16.0, 16.0, 0.0, 0.0, 16.0, -16.0])
    assert loss == pytest.approx(64.0)  # 8**2
    assert grad.sum() == pytest.approx(0.0)  # opposite signs cancel


def test_fit_on_exact_tree_metric_is_a_noop():
    d = line_metric([0, 1, 3, 7, 12, 20])
    w, x, history = fit_tropical_gradient_descent(d, CONFIG)
    np.testing.assert_allclose(w, 0.0, atol=1e-12)  # nothing to correct
    np.testing.assert_allclose(x, d, atol=1e-9)
    assert tropical_score(x)["l2_loss"] < 1e-12
    assert len(history) >= 1
    assert tuple(history.columns) == HISTORY_COLUMNS


def test_fit_reduces_violations_on_noisy_matrix():
    d = noisy_tree_metric(seed=0)
    before = tropical_score(d)["l2_loss"]
    _, x, history = fit_tropical_gradient_descent(d, CONFIG)
    after = tropical_score(x)["l2_loss"]
    assert before > 0.0
    assert after < before  # violations decreased overall
    # overall (not necessarily monotone) loss decrease
    assert history["loss"].iloc[-1] <= history["loss"].iloc[0]


def test_fit_preserves_distance_matrix_invariants():
    d = noisy_tree_metric(seed=1)
    result = correct_distance_matrix(d, CONFIG)
    x, omega = result["X"], result["omega"]
    assert is_valid_distance_matrix(x)  # symmetric, zero diag, non-negative
    np.testing.assert_allclose(omega, omega.T)  # omega symmetric
    np.testing.assert_allclose(np.diag(omega), 0.0)  # zero diagonal


def test_correct_distance_matrix_reports_positive_improvement():
    d = noisy_tree_metric(seed=2)
    result = correct_distance_matrix(d, CONFIG)
    assert result["relative_improvement"] > 0.0
    assert result["metrics_after"]["l2_loss"] < result["metrics_before"]["l2_loss"]
    assert set(result) == {
        "omega",
        "omega_vector",
        "X",
        "metrics_before",
        "metrics_after",
        "relative_improvement",
        "history",
    }


def test_fit_is_deterministic():
    d = noisy_tree_metric(seed=3)
    w1, _, _ = fit_tropical_gradient_descent(d, CONFIG)
    w2, _, _ = fit_tropical_gradient_descent(d, CONFIG)
    np.testing.assert_array_equal(w1, w2)
