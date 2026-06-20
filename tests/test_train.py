"""Tests for :mod:`src.train` / :mod:`src.evaluate` / :mod:`src.visualization`.

This doubles as the end-to-end smoke test on the bundled synthetic demo data.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src.evaluate import generate_report
from src.train import run_training
from src.utils import load_params
from src.visualization import (
    plot_before_after_summary,
    plot_loss_curve,
    plot_violation_histograms,
)

DEMO_CSV = "data/sample/sample_viral_sequences.csv"


def _demo_params(tmp_path: Path) -> dict:
    params = copy.deepcopy(load_params())
    params["data"]["min_seq_length"] = 1  # demo sequences are short
    params["optimization"]["epochs"] = 40
    params["outputs"] = {
        "output_dir": str(tmp_path / "outputs"),
        "figures_dir": str(tmp_path / "outputs" / "figures"),
        "trees_dir": str(tmp_path / "outputs" / "trees"),
        "models_dir": str(tmp_path / "models"),
    }
    return params


def test_training_produces_all_artifacts(tmp_path):
    params = _demo_params(tmp_path)
    summary = run_training(params=params, input_path=DEMO_CSV, enable_mlflow=False)

    assert summary["n_sequences"] == 12
    assert summary["relative_improvement"] >= 0.0

    out = Path(params["outputs"]["output_dir"])
    for name in (
        "distance_matrix.csv",
        "corrected_distance_matrix.csv",
        "omega.csv",
        "history.csv",
        "metrics.json",
    ):
        assert (out / name).exists(), name
    trees = Path(params["outputs"]["trees_dir"])
    for name in ("tree_after.newick", "tree_after_edges.csv", "tree_after.dot"):
        assert (trees / name).exists(), name
    figs = Path(params["outputs"]["figures_dir"])
    for name in (
        "loss_curve.png",
        "violation_histograms.png",
        "before_after_summary.png",
    ):
        assert (figs / name).exists(), name
    assert (Path(params["outputs"]["models_dir"]) / "omega.npy").exists()


def test_evaluate_report_renders(tmp_path):
    params = _demo_params(tmp_path)
    run_training(params=params, input_path=DEMO_CSV, enable_mlflow=False)
    report = generate_report(
        output_dir=params["outputs"]["output_dir"],
        report_path=tmp_path / "report.md",
    )
    assert "Evaluation Report" in report
    assert "relative improvement" in report.lower()
    assert (tmp_path / "report.md").exists()


def test_training_missing_input_raises(tmp_path):
    params = _demo_params(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_training(
            params=params, input_path=tmp_path / "missing.csv", enable_mlflow=False
        )


def test_evaluate_without_metrics_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_report(output_dir=tmp_path, report_path=tmp_path / "r.md")


def test_visualization_functions_return_figures(tmp_path):
    import pandas as pd

    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "loss": [3.0, 2.0, 1.0],
            "grad_tr_norm": [1, 1, 1],
            "eta": [0.1, 0.1, 0.1],
        }
    )
    f1 = plot_loss_curve(history, tmp_path / "loss.png")
    f2 = plot_violation_histograms(
        np.array([0.5, 0.3, 0.1]), np.array([0.1, 0.05, 0.0]), tmp_path / "hist.png"
    )
    f3 = plot_before_after_summary(
        {"mean_violation": 0.3, "max_violation": 0.5, "l2_loss": 1.0},
        {"mean_violation": 0.05, "max_violation": 0.1, "l2_loss": 0.1},
        tmp_path / "sum.png",
    )
    for fig, name in ((f1, "loss.png"), (f2, "hist.png"), (f3, "sum.png")):
        assert fig is not None
        assert (tmp_path / name).exists()
