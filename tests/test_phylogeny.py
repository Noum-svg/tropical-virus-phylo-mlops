"""Tests for :mod:`src.phylogeny` -- Neighbor-Joining and tree export."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import pytest
from Bio import Phylo

from src.phylogeny import neighbor_joining, reconstruct_tree


def line_metric(points) -> np.ndarray:
    """Additive (tree) metric ``d_ij = |x_i - x_j|`` -- exactly reconstructible."""
    x = np.asarray(points, dtype=float)
    return np.abs(x[:, None] - x[None, :])


def parse(newick: str):
    return Phylo.read(StringIO(newick), "newick")


def test_newick_parses_and_contains_every_label_once():
    labels = ["Virus_A", "Virus_B", "Virus_C", "Virus_D", "Virus_E"]
    tree = neighbor_joining(line_metric([0, 2, 5, 9, 14]), labels)
    parsed = parse(tree.to_newick())
    terminals = sorted(t.name for t in parsed.get_terminals())
    assert terminals == sorted(labels)
    assert len(parsed.get_terminals()) == len(labels)


def test_edge_list_and_dot_are_consistent():
    labels = ["a", "b", "c", "d", "e", "f"]
    tree = neighbor_joining(line_metric([0, 1, 3, 6, 10, 15]), labels)

    edf = tree.to_edge_dataframe()
    assert list(edf.columns) == ["parent", "child", "branch_length"]
    assert len(edf) == 2 * len(labels) - 3  # edge count of an unrooted binary tree
    assert (edf["branch_length"] >= 0).all()  # clamped for display
    leaf_children = sorted(c for c in edf["child"] if c in labels)
    assert leaf_children == sorted(labels)  # each leaf appears once as a child

    assert len(tree.to_edge_list()) == len(edf)
    dot = tree.to_dot()
    assert dot.startswith("digraph")
    assert "->" in dot
    for label in labels:
        assert f'"{label}"' in dot


def test_nj_recovers_additive_metric_path_distances():
    labels = ["t0", "t1", "t2", "t3", "t4", "t5"]
    d = line_metric([0, 2, 5, 9, 14, 20])
    tree = neighbor_joining(d, labels)
    parsed = parse(tree.to_newick())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            assert parsed.distance(labels[i], labels[j]) == pytest.approx(
                d[i, j], abs=1e-6
            )


def test_reconstruct_from_dataframe_uses_index():
    labels = ["x", "y", "z", "w"]
    frame = pd.DataFrame(line_metric([0, 3, 7, 12]), index=labels, columns=labels)
    tree = reconstruct_tree(frame)
    assert set(tree.labels) == set(labels)
    terminals = sorted(t.name for t in parse(tree.to_newick()).get_terminals())
    assert terminals == sorted(labels)


def test_tiny_inputs():
    one = neighbor_joining(np.zeros((1, 1)), ["solo"])
    assert one.to_newick() == "solo;"
    assert one.to_edge_list() == []

    two = neighbor_joining(np.array([[0.0, 0.4], [0.4, 0.0]]), ["p", "q"])
    assert sorted(t.name for t in parse(two.to_newick()).get_terminals()) == ["p", "q"]

    three = neighbor_joining(line_metric([0, 1, 3]), ["u", "v", "w"])
    assert sorted(t.name for t in parse(three.to_newick()).get_terminals()) == [
        "u",
        "v",
        "w",
    ]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        neighbor_joining(np.zeros((3, 3)), ["a", "b"])  # label count mismatch
    with pytest.raises(ValueError):
        neighbor_joining(np.zeros((2, 3)), ["a", "b"])  # not square


def test_labels_with_special_characters_round_trip():
    labels = ["Virus A", "Strain(1)", "iso:2", "v,3"]
    tree = neighbor_joining(line_metric([0, 4, 9, 15]), labels)
    terminals = sorted(t.name for t in parse(tree.to_newick()).get_terminals())
    assert terminals == sorted(labels)
