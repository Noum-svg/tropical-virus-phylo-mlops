"""Batch training entry point: build D, learn omega, reconstruct the tree.

Writes all artifacts under ``outputs/`` (matrices, omega, history, metrics,
figures, trees) and ``models/omega.npy``, and logs the run to MLflow (best
effort; a missing or failing MLflow never breaks training).

Run with::

    python -m src.train
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import load_and_clean
from src.distances import build_distance_matrix
from src.phylogeny import reconstruct_tree
from src.tropical_gradient_descent import correct_distance_matrix
from src.tropical_grassmannian import compute_violations
from src.utils import generate_quadruplets, load_params
from src.visualization import (
    plot_before_after_summary,
    plot_loss_curve,
    plot_violation_histograms,
)


def _log_to_mlflow(
    params: dict[str, Any],
    flat_metrics: dict[str, float],
    artifacts: list[Path],
) -> None:
    """Log params, metrics, and artifacts to MLflow (best effort)."""
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - mlflow optional at runtime
        print(f"[train] MLflow not available, skipping logging: {exc}")
        return
    try:
        ml_cfg = params.get("mlflow", {})
        if ml_cfg.get("tracking_uri"):
            mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
        mlflow.set_experiment(ml_cfg.get("experiment_name", "tropical-virus-phylo"))
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "dataset_path": params["data"]["raw_path"],
                    "min_seq_length": params["data"]["min_seq_length"],
                    "alpha": params["distance"]["alpha"],
                    "epochs": params["optimization"]["epochs"],
                    "gamma": params["optimization"]["gamma"],
                    "lambda_reg": params["optimization"]["lambda_reg"],
                    "epsilon": params["optimization"]["epsilon"],
                    "quadruplet_sample_size": params["optimization"][
                        "quadruplet_sample_size"
                    ],
                    "seed": params.get("seed", 42),
                }
            )
            mlflow.log_metrics(flat_metrics)
            for art in artifacts:
                if art.exists():
                    mlflow.log_artifact(str(art))
    except Exception as exc:  # pragma: no cover - logging must never break training
        print(f"[train] MLflow logging failed (continuing): {exc}")


def run_training(
    params: dict[str, Any] | None = None,
    input_path: str | Path | None = None,
    matrix_path: str | Path | None = None,
    enable_mlflow: bool = True,
    write_tree: bool = True,
) -> dict[str, Any]:
    """Execute the full training pipeline and persist artifacts.

    Parameters
    ----------
    params : dict or None, optional
        Configuration; loaded from ``params.yaml`` when ``None``.
    input_path : str or pathlib.Path or None, optional
        Override the input CSV path (defaults to ``params['data']['raw_path']``).
    matrix_path : str or pathlib.Path or None, optional
        If given, load the distance matrix ``D`` directly from this CSV (indexed
        by virus name) instead of building it from sequences. Used by the DVC
        ``tropical_correction`` stage so it depends on the staged matrix.
    enable_mlflow : bool, optional
        Log the run to MLflow when ``True``.
    write_tree : bool, optional
        Write the Newick/edges/DOT tree artifacts. The DVC pipeline sets this
        ``False`` so the separate ``phylogenetic_tree`` stage owns those files.

    Returns
    -------
    dict
        Summary with ``n_sequences``, ``relative_improvement``,
        ``runtime_seconds``, ``output_dir``, and ``tree_newick``.

    Raises
    ------
    FileNotFoundError
        If the input CSV is absent (sequences are never fabricated).
    """
    params = params or load_params()
    seed = int(params.get("seed", 42))
    data_cfg, dist_cfg, opt_cfg, out_cfg = (
        params["data"],
        params["distance"],
        params["optimization"],
        params["outputs"],
    )
    input_path = input_path or data_cfg["raw_path"]

    start = time.time()
    if matrix_path is not None:
        distance_matrix = pd.read_csv(matrix_path, index_col=0)
        distance_matrix.columns = distance_matrix.index
        names = list(distance_matrix.index)
    else:
        clean_df = load_and_clean(input_path, min_seq_length=data_cfg["min_seq_length"])
        names = list(clean_df["virus_name"])
        distance_matrix = build_distance_matrix(
            list(clean_df["clean_sequence"]), names, alpha=dist_cfg["alpha"]
        )
    n = len(names)

    config = {
        "epochs": opt_cfg["epochs"],
        "gamma": opt_cfg["gamma"],
        "lambda_reg": opt_cfg["lambda_reg"],
        "epsilon": opt_cfg["epsilon"],
        "quadruplet_sample_size": opt_cfg["quadruplet_sample_size"],
        "seed": seed,
    }
    correction = correct_distance_matrix(distance_matrix.to_numpy(), config)
    corrected = pd.DataFrame(correction["X"], index=names, columns=names)
    omega = pd.DataFrame(correction["omega"], index=names, columns=names)
    tree = reconstruct_tree(corrected)

    quads = generate_quadruplets(n, config["quadruplet_sample_size"], seed)
    deltas_before = compute_violations(distance_matrix.to_numpy(), quads)
    deltas_after = compute_violations(correction["X"], quads)
    runtime = time.time() - start

    # --- persist artifacts ---
    out_dir = Path(out_cfg["output_dir"])
    fig_dir = Path(out_cfg["figures_dir"])
    tree_dir = Path(out_cfg["trees_dir"])
    model_dir = Path(out_cfg["models_dir"])
    for d in (out_dir, fig_dir, tree_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    distance_matrix.to_csv(out_dir / "distance_matrix.csv")
    corrected.to_csv(out_dir / "corrected_distance_matrix.csv")
    omega.to_csv(out_dir / "omega.csv")
    correction["history"].to_csv(out_dir / "history.csv", index=False)
    np.save(model_dir / "omega.npy", correction["omega"])

    metrics = {
        "n_sequences": n,
        "relative_improvement": correction["relative_improvement"],
        "runtime_seconds": runtime,
        "metrics_before": correction["metrics_before"],
        "metrics_after": correction["metrics_after"],
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    if write_tree:
        (tree_dir / "tree_after.newick").write_text(tree.to_newick(), encoding="utf-8")
        tree.to_edge_dataframe().to_csv(tree_dir / "tree_after_edges.csv", index=False)
        (tree_dir / "tree_after.dot").write_text(tree.to_dot(), encoding="utf-8")

    loss_fig = fig_dir / "loss_curve.png"
    hist_fig = fig_dir / "violation_histograms.png"
    summary_fig = fig_dir / "before_after_summary.png"
    plot_loss_curve(correction["history"], loss_fig)
    plot_violation_histograms(deltas_before, deltas_after, hist_fig)
    plot_before_after_summary(
        correction["metrics_before"], correction["metrics_after"], summary_fig
    )
    import matplotlib.pyplot as plt  # backend already set to Agg in visualization

    plt.close("all")

    if enable_mlflow:
        before, after = correction["metrics_before"], correction["metrics_after"]
        flat_metrics = {
            "l2_before": before["l2_loss"],
            "l2_after": after["l2_loss"],
            "relative_improvement": correction["relative_improvement"],
            "mean_violation_before": before["mean_violation"],
            "mean_violation_after": after["mean_violation"],
            "max_violation_before": before["max_violation"],
            "max_violation_after": after["max_violation"],
            "percent_exact_before": before["percent_exact"],
            "percent_exact_after": after["percent_exact"],
            "runtime_seconds": runtime,
        }
        artifacts = [
            out_dir / "distance_matrix.csv",
            out_dir / "corrected_distance_matrix.csv",
            out_dir / "omega.csv",
            out_dir / "history.csv",
            out_dir / "metrics.json",
            tree_dir / "tree_after.newick",
            tree_dir / "tree_after_edges.csv",
            tree_dir / "tree_after.dot",
            loss_fig,
            hist_fig,
            summary_fig,
        ]
        _log_to_mlflow(params, flat_metrics, artifacts)

    return {
        "n_sequences": n,
        "relative_improvement": correction["relative_improvement"],
        "runtime_seconds": runtime,
        "output_dir": str(out_dir),
        "tree_newick": tree.to_newick(),
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Train: build D, learn omega, tree.")
    parser.add_argument("--input", default=None, help="Input CSV path.")
    parser.add_argument(
        "--matrix", default=None, help="Precomputed distance matrix CSV."
    )
    parser.add_argument("--no-tree", action="store_true", help="Skip tree artifacts.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow.")
    args = parser.parse_args()
    summary = run_training(
        input_path=args.input,
        matrix_path=args.matrix,
        enable_mlflow=not args.no_mlflow,
        write_tree=not args.no_tree,
    )
    print(json.dumps(summary, indent=2))
