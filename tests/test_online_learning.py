"""Tests for :mod:`src.online_learning` -- warm start, incremental taxa, persistence.

Toy sequences are explicit unit-test inputs, not scientific results.
"""

from __future__ import annotations

from io import StringIO

import numpy as np
from Bio import Phylo

from src.online_learning import OnlineTropicalModel

NAMES = ["a", "b", "c", "d", "e", "f"]
SEQS = [
    "ACGTACGTACGTACGTACGT",
    "ACGTACGAACGTACGTACGT",
    "ACGTTCGTACGTACGTTCGT",
    "TTTTACGTACGTACGTTTTT",
    "ACGTACGTGGGTACGTACGT",
    "ACGTACGTACGTAAGTACGT",
]
CFG = {"epochs": 80, "quadruplet_sample_size": None}


def test_fit_then_partial_fit_does_not_worsen():
    model = OnlineTropicalModel(config=CFG).fit(NAMES, SEQS)
    ri1 = model.score()["relative_improvement"]
    model.partial_fit(epochs=80)  # warm start, more epochs
    ri2 = model.score()["relative_improvement"]
    assert ri2 >= ri1 - 1e-6  # continued training never worsens the correction
    assert ri2 >= -1e-9


def test_add_sequences_grows_model_online():
    model = OnlineTropicalModel(config=CFG).fit(NAMES[:4], SEQS[:4])
    assert model.n == 4
    omega_small = model.omega.copy()

    model.add_sequences(NAMES[4:], SEQS[4:], epochs=60)  # online update
    assert model.n == 6
    assert model.omega.shape[0] == 6 * 5 // 2  # enlarged pair space

    score = model.score()
    assert score["relative_improvement"] >= -1e-9
    tree = Phylo.read(StringIO(model.tree().to_newick()), "newick")
    assert sorted(t.name for t in tree.get_terminals()) == sorted(NAMES)
    # the small model's omega was strictly smaller
    assert omega_small.shape[0] == 4 * 3 // 2


def test_add_sequences_on_empty_behaves_like_fit():
    model = OnlineTropicalModel(config={"epochs": 40, "quadruplet_sample_size": None})
    model.add_sequences(NAMES, SEQS)
    assert model.n == 6
    assert model.score()["relative_improvement"] >= -1e-9


def test_save_load_roundtrip_and_resume(tmp_path):
    model = OnlineTropicalModel(config=CFG).fit(NAMES, SEQS)
    path = model.save(tmp_path / "model.npz")
    assert path.exists()

    loaded = OnlineTropicalModel.load(tmp_path / "model.npz")
    assert loaded.names == model.names
    assert loaded.sequences == model.sequences
    np.testing.assert_allclose(loaded.omega, model.omega)
    # a loaded model can score and resume online learning
    assert loaded.score()["relative_improvement"] >= -1e-9
    loaded.partial_fit(epochs=20)
    assert loaded.n == 6


def test_history_accumulates_across_updates():
    model = OnlineTropicalModel(config=CFG).fit(NAMES, SEQS)
    rows_after_fit = len(model.history)
    model.partial_fit(epochs=30)
    assert len(model.history) > rows_after_fit  # history grows with continued use
