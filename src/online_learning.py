"""Online (incremental) learning for the tropical correction.

The "model" is the symmetric correction ``omega``. In normal use the model can
keep improving instead of being retrained from scratch:

* :meth:`OnlineTropicalModel.partial_fit` continues Tropical Gradient Descent
  from the current ``omega`` (warm start) for more epochs;
* :meth:`OnlineTropicalModel.add_sequences` incorporates newly observed taxa: it
  expands ``omega`` into the larger pair space (new pairs start at zero) and
  warm-starts continued training — far cheaper than retraining on everything;
* :meth:`OnlineTropicalModel.save` / :meth:`load` persist the model so it
  survives across sessions and can be resumed.

Because the optimizer returns the best iterate (which always includes the warm
start), continued training never makes the correction worse than the warm start.
This module orchestrates :mod:`src` only -- no new math.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import clean_sequence
from src.distances import build_distance_matrix
from src.phylogeny import reconstruct_tree
from src.tropical_gradient_descent import fit_tropical_gradient_descent
from src.tropical_grassmannian import tropical_score
from src.utils import matrix_to_vector, project_to_distance_space, vector_to_matrix

DEFAULT_CONFIG: dict[str, Any] = {
    "epochs": 200,
    "gamma": 0.05,
    "lambda_reg": 0.01,
    "epsilon": 1e-8,
    "quadruplet_sample_size": 5000,
    "seed": 42,
}


class OnlineTropicalModel:
    """An incrementally-trainable tropical correction model.

    Parameters
    ----------
    alpha : float, optional
        Distance Hamming/length weighting. Defaults to ``0.9``.
    config : dict or None, optional
        Optimization configuration (merged over :data:`DEFAULT_CONFIG`).
    """

    def __init__(self, alpha: float = 0.9, config: dict[str, Any] | None = None):
        self.alpha = float(alpha)
        self.config: dict[str, Any] = {**DEFAULT_CONFIG, **(config or {})}
        self.names: list[str] = []
        self.sequences: list[str] = []  # cleaned ACGT sequences
        self.omega: np.ndarray = np.zeros(0, dtype=float)
        self.history: pd.DataFrame = pd.DataFrame()

    @property
    def n(self) -> int:
        """Number of taxa currently in the model."""
        return len(self.names)

    def _distance_matrix(self) -> np.ndarray:
        return build_distance_matrix(
            self.sequences, self.names, alpha=self.alpha
        ).to_numpy()

    def _run(self, epochs: int | None, init_omega: np.ndarray | None) -> None:
        cfg = dict(self.config)
        if epochs is not None:
            cfg["epochs"] = int(epochs)
        if init_omega is not None:
            cfg["init_omega"] = init_omega
        omega, _x, hist = fit_tropical_gradient_descent(self._distance_matrix(), cfg)
        self.omega = omega
        self.history = (
            hist
            if self.history.empty
            else pd.concat([self.history, hist], ignore_index=True)
        )

    def fit(
        self, names: list[str], sequences: list[str], epochs: int | None = None
    ) -> "OnlineTropicalModel":
        """Train from scratch (``omega`` starts at zero).

        Parameters
        ----------
        names : list of str
            Taxon names.
        sequences : list of str
            Raw sequences (cleaned internally).
        epochs : int or None, optional
            Override the configured epoch count.
        """
        if len(names) != len(sequences):
            raise ValueError("names and sequences must have the same length.")
        self.names = [str(x) for x in names]
        self.sequences = [clean_sequence(s) for s in sequences]
        self.history = pd.DataFrame()
        self._run(epochs, init_omega=None)
        return self

    def partial_fit(self, epochs: int | None = None) -> "OnlineTropicalModel":
        """Continue training from the current ``omega`` (warm start)."""
        if self.n < 2:
            raise ValueError("Call fit() with at least two sequences first.")
        self._run(epochs, init_omega=self.omega)
        return self

    def add_sequences(
        self, names: list[str], sequences: list[str], epochs: int | None = None
    ) -> "OnlineTropicalModel":
        """Incorporate new taxa and warm-start continued training (online update).

        The existing ``omega`` is embedded into the enlarged pair space (new pairs
        initialized to zero), then the optimizer resumes from there.
        """
        if len(names) != len(sequences):
            raise ValueError("names and sequences must have the same length.")
        if self.n == 0:
            return self.fit(names, sequences, epochs=epochs)

        old_n = self.n
        old_omega_mat = vector_to_matrix(self.omega, old_n)
        self.names += [str(x) for x in names]
        self.sequences += [clean_sequence(s) for s in sequences]

        big = np.zeros((self.n, self.n), dtype=float)
        big[:old_n, :old_n] = old_omega_mat
        self._run(epochs, init_omega=matrix_to_vector(big))
        return self

    def corrected_matrix(self) -> np.ndarray:
        """Return the corrected, projected matrix ``X = Pi(D + omega)``."""
        d = self._distance_matrix()
        if self.n < 2:
            return d
        return project_to_distance_space(d + vector_to_matrix(self.omega, self.n))

    def score(self) -> dict[str, Any]:
        """Before/after tropical metrics and relative improvement on the same sample."""
        d = self._distance_matrix()
        x = self.corrected_matrix()
        eps = self.config["epsilon"]
        ss = self.config["quadruplet_sample_size"]
        seed = self.config["seed"]
        before = tropical_score(d, eps=eps, sample_size=ss, seed=seed)
        after = tropical_score(x, eps=eps, sample_size=ss, seed=seed)
        ri = (before["l2_loss"] - after["l2_loss"]) / (before["l2_loss"] + eps)
        return {
            "metrics_before": before,
            "metrics_after": after,
            "relative_improvement": float(ri),
        }

    def tree(self):
        """Reconstruct the Neighbor-Joining tree from the corrected matrix."""
        names = self.names
        frame = pd.DataFrame(self.corrected_matrix(), index=names, columns=names)
        return reconstruct_tree(frame)

    def save(self, path: str | Path) -> Path:
        """Persist the model (omega, taxa, sequences, config) to a ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            omega=self.omega,
            names=np.array(self.names, dtype=object),
            sequences=np.array(self.sequences, dtype=object),
            alpha=np.array(self.alpha),
            config=np.array(json.dumps(self.config)),
        )
        return path if path.suffix else path.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> "OnlineTropicalModel":
        """Load a model previously saved with :meth:`save`."""
        data = np.load(Path(path), allow_pickle=True)
        model = cls(alpha=float(data["alpha"]), config=json.loads(str(data["config"])))
        model.names = [str(x) for x in data["names"]]
        model.sequences = [str(x) for x in data["sequences"]]
        model.omega = np.asarray(data["omega"], dtype=float)
        return model
