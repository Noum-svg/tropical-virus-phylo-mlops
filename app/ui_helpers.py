"""Plotting helpers for the Streamlit dashboard.

These build matplotlib figures only; all scientific computation stays in
``src/``. Loss/violation/summary figures are reused from
:mod:`src.visualization`; this module adds heatmaps, distributions, and tree
renderings (rectangular via Biopython, circular via a small radial layout).
"""

from __future__ import annotations

import math
from io import StringIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from Bio import Phylo  # noqa: E402

from src.phylogeny import PhyloTree  # noqa: E402

# Re-exported so the app imports all figure helpers from one place.
from src.visualization import (  # noqa: E402,F401
    plot_before_after_summary,
    plot_loss_curve,
    plot_violation_histograms,
)


def heatmap(matrix: pd.DataFrame, title: str = "Matrix") -> "plt.Figure":
    """Render a labeled heatmap of a (distance) matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    data = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    im = ax.imshow(data, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if isinstance(matrix, pd.DataFrame) and data.shape[0] <= 30:
        ax.set_xticks(range(data.shape[0]))
        ax.set_yticks(range(data.shape[0]))
        ax.set_xticklabels(matrix.index, rotation=90, fontsize=7)
        ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def distribution(
    values, title: str = "Distribution", xlabel: str = "value"
) -> "plt.Figure":
    """Histogram of a 1-D set of values (e.g. distances or violations)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size:
        ax.hist(arr, bins=30, color="#1f77b4", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def draw_tree_rectangular(newick: str) -> "plt.Figure":
    """Draw a rectangular cladogram/phylogram from a Newick string."""
    fig, ax = plt.subplots(figsize=(7, 6))
    tree = Phylo.read(StringIO(newick), "newick")
    Phylo.draw(tree, axes=ax, do_show=False)
    ax.set_title("Phylogenetic tree (rectangular)")
    fig.tight_layout()
    return fig


def _radial_layout(tree: PhyloTree) -> dict[str, tuple[float, float]]:
    """Compute (x, y) positions for a circular/radial tree layout."""
    children: dict[str, list[tuple[str, float]]] = {}
    for e in tree.edges:
        children.setdefault(e["parent"], []).append((e["child"], e["branch_length"]))

    leaves: list[str] = []

    def collect(node: str) -> None:
        kids = children.get(node, [])
        if not kids:
            leaves.append(node)
        for child, _ in kids:
            collect(child)

    collect(tree.root)
    n_leaves = max(len(leaves), 1)
    leaf_angle = {leaf: 2 * math.pi * i / n_leaves for i, leaf in enumerate(leaves)}

    radius: dict[str, float] = {tree.root: 0.0}
    angle: dict[str, float] = {}

    def assign(node: str) -> float:
        kids = children.get(node, [])
        if not kids:
            angle[node] = leaf_angle[node]
            return angle[node]
        child_angles = []
        for child, length in kids:
            radius[child] = radius.get(node, 0.0) + max(length, 1e-6)
            child_angles.append(assign(child))
        angle[node] = sum(child_angles) / len(child_angles)
        return angle[node]

    assign(tree.root)
    return {
        node: (radius.get(node, 0.0) * math.cos(a), radius.get(node, 0.0) * math.sin(a))
        for node, a in angle.items()
    }


def draw_tree_circular(tree: PhyloTree) -> "plt.Figure":
    """Draw a circular/radial layout of a :class:`PhyloTree`."""
    pos = _radial_layout(tree)
    leafset = set(tree.labels)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for e in tree.edges:
        if e["parent"] in pos and e["child"] in pos:
            x0, y0 = pos[e["parent"]]
            x1, y1 = pos[e["child"]]
            ax.plot([x0, x1], [y0, y1], color="#888888", linewidth=1.0, zorder=1)
    for node, (x, y) in pos.items():
        if node in leafset:
            ax.scatter([x], [y], s=18, color="#2ca02c", zorder=2)
            ax.text(x, y, f" {node}", fontsize=7, va="center")
        else:
            ax.scatter([x], [y], s=8, color="#1f77b4", zorder=2)
    ax.set_title("Phylogenetic tree (circular)")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig
