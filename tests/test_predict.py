"""Tests for :mod:`src.predict` -- end-to-end orchestration.

Sequences here are small explicit toy unit-test inputs, not scientific results.
"""

from __future__ import annotations

from io import StringIO

import pytest
from Bio import Phylo

from src.predict import predict_from_sequences

TOY_NAMES = ["v1", "v2", "v3", "v4", "v5"]
TOY_SEQS = [
    "AUGGUGAAGUACCUAACGUAGCUAGCUA",
    "AUGGUGAAAUAUCUAACGUAGCUAGCUA",
    "AUGGUGAAGUACCUAACGUAGGUAGCUA",
    "ACGUACGUACGUACGUACGUACGUACGU",
    "ACGUACGUACGAACGUACGUUCGUACGU",
]


def test_predict_returns_full_payload():
    result = predict_from_sequences(
        TOY_NAMES, TOY_SEQS, epochs=50, quadruplet_sample_size=None, seed=42
    )
    assert set(result) == {
        "virus_names",
        "clean_sequences",
        "sequence_lengths",
        "distance_matrix",
        "corrected_distance_matrix",
        "omega",
        "metrics_before",
        "metrics_after",
        "relative_improvement",
        "tree",
        "tree_newick",
        "tree_edges",
        "tree_dot",
        "history",
    }
    n = len(TOY_NAMES)
    assert result["virus_names"] == TOY_NAMES
    assert result["distance_matrix"].shape == (n, n)
    assert result["corrected_distance_matrix"].shape == (n, n)
    assert list(result["distance_matrix"].index) == TOY_NAMES


def test_predict_tree_parses_and_has_all_taxa():
    result = predict_from_sequences(TOY_NAMES, TOY_SEQS, epochs=50)
    parsed = Phylo.read(StringIO(result["tree_newick"]), "newick")
    terminals = sorted(t.name for t in parsed.get_terminals())
    assert terminals == sorted(TOY_NAMES)
    assert len(result["tree_edges"]) == 2 * len(TOY_NAMES) - 3


def test_predict_reports_improvement_and_clean_sequences():
    result = predict_from_sequences(TOY_NAMES, TOY_SEQS, epochs=200, seed=42)
    assert result["relative_improvement"] >= 0.0
    # cleaning converted U -> T everywhere
    assert all("U" not in s for s in result["clean_sequences"])
    assert all(set(s) <= set("ACGT") for s in result["clean_sequences"])


def test_predict_length_mismatch_raises():
    with pytest.raises(ValueError):
        predict_from_sequences(["a", "b"], ["ACGT"])


def test_predict_all_invalid_raises():
    with pytest.raises(ValueError):
        predict_from_sequences(["a", "b"], ["NNNN", "----"], min_seq_length=1)
