"""Matplotlib figures for the pipeline (loss curve, violation histograms, summary).

All functions use the non-interactive ``Agg`` backend so they work headless (CI,
servers). Each returns the :class:`matplotlib.figure.Figure` and optionally saves
it to ``path``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig: "plt.Figure", path: str | Path | None) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")


def plot_loss_curve(
    history: pd.DataFrame, path: str | Path | None = None
) -> "plt.Figure":
    """Plot the training loss against epochs.

    Parameters
    ----------
    history : pandas.DataFrame
        Must contain ``epoch`` and ``loss`` columns.
    path : str or pathlib.Path or None, optional
        If given, save the figure there.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(history) > 0:
        ax.plot(history["epoch"], history["loss"], color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Tropical Gradient Descent loss")
    ax.grid(True, alpha=0.3)
    _save(fig, path)
    return fig


def plot_violation_histograms(
    deltas_before: np.ndarray,
    deltas_after: np.ndarray,
    path: str | Path | None = None,
    bins: int = 30,
) -> "plt.Figure":
    """Overlay histograms of four-point violations before vs after correction.

    Parameters
    ----------
    deltas_before, deltas_after : numpy.ndarray
        Per-quadruplet violations of ``D`` and ``X``.
    path : str or pathlib.Path or None, optional
        If given, save the figure there.
    bins : int, optional
        Histogram bin count.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    before = np.asarray(deltas_before, dtype=float)
    after = np.asarray(deltas_after, dtype=float)
    if before.size or after.size:
        lo = 0.0
        hi = float(max(before.max(initial=0.0), after.max(initial=0.0), 1e-9))
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(before, bins=edges, alpha=0.6, label="before", color="#d62728")
        ax.hist(after, bins=edges, alpha=0.6, label="after", color="#2ca02c")
        ax.legend()
    ax.set_xlabel("Four-point violation (delta)")
    ax.set_ylabel("Count")
    ax.set_title("Violation distribution: before vs after")
    ax.grid(True, alpha=0.3)
    _save(fig, path)
    return fig


def plot_before_after_summary(
    metrics_before: dict,
    metrics_after: dict,
    path: str | Path | None = None,
) -> "plt.Figure":
    """Grouped bar chart comparing key metrics before vs after correction.

    Parameters
    ----------
    metrics_before, metrics_after : dict
        Outputs of :func:`src.tropical_grassmannian.tropical_score`.
    path : str or pathlib.Path or None, optional
        If given, save the figure there.

    Returns
    -------
    matplotlib.figure.Figure
    """
    keys = ["mean_violation", "max_violation", "l2_loss"]
    before = [float(metrics_before.get(k, 0.0)) for k in keys]
    after = [float(metrics_after.get(k, 0.0)) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(keys))
    width = 0.38
    ax.bar(x - width / 2, before, width, label="before", color="#d62728")
    ax.bar(x + width / 2, after, width, label="after", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=15)
    ax.set_ylabel("Value")
    ax.set_title("Tropical metrics: before vs after")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, path)
    return fig


# --------------------------------------------------------------------------- #
# Phylogenetic tree rendering (clade-coloured dendrogram)
# --------------------------------------------------------------------------- #
_CLADE_PALETTE = [
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#9333ea",
    "#dc2626",
    "#0891b2",
    "#ca8a04",
    "#db2777",
    "#0d9488",
    "#7c3aed",
]


def _tree_geometry(tree, use_branch_length: bool):
    """Return children map, leaf order, depth (x), y position, and clade colour."""
    children: dict[str, list[tuple[str, float]]] = {}
    for e in tree.edges:
        children.setdefault(e["parent"], []).append(
            (e["child"], float(e["branch_length"]))
        )
    root = tree.root

    leaves: list[str] = []

    def dfs(node: str) -> None:
        kids = children.get(node, [])
        if not kids:
            leaves.append(node)
        for child, _ in kids:
            dfs(child)

    dfs(root)
    leaf_index = {lf: i for i, lf in enumerate(leaves)}

    depth: dict[str, float] = {root: 0.0}

    def set_depth(node: str) -> None:
        for child, bl in children.get(node, []):
            depth[child] = depth[node] + (max(bl, 1e-6) if use_branch_length else 1.0)
            set_depth(child)

    set_depth(root)

    ypos: dict[str, float] = {}

    def set_y(node: str) -> float:
        kids = children.get(node, [])
        if not kids:
            ypos[node] = float(leaf_index[node])
            return ypos[node]
        ys = [set_y(c) for c, _ in kids]
        ypos[node] = sum(ys) / len(ys)
        return ypos[node]

    set_y(root)

    color: dict[str, str] = {root: "#94a3b8"}

    def paint(node: str, col: str) -> None:
        color[node] = col
        for child, _ in children.get(node, []):
            paint(child, col)

    for i, (child, _) in enumerate(children.get(root, [])):
        paint(child, _CLADE_PALETTE[i % len(_CLADE_PALETTE)])

    return children, leaves, depth, ypos, color


def render_tree(
    tree,
    layout: str = "circular",
    show_labels: bool = True,
    label_size: int = 8,
    use_branch_length: bool = True,
    path: str | Path | None = None,
) -> "plt.Figure":
    """Render a :class:`src.phylogeny.PhyloTree` as a clade-coloured dendrogram.

    Parameters
    ----------
    tree : PhyloTree
        The reconstructed tree.
    layout : {"circular", "rectangular"}, optional
        Drawing layout.
    show_labels : bool, optional
        Draw leaf labels.
    label_size : int, optional
        Leaf label font size.
    use_branch_length : bool, optional
        Use branch lengths for the radial/horizontal extent (phylogram) instead
        of topological depth (cladogram).
    path : str or pathlib.Path or None, optional
        Save location.

    Returns
    -------
    matplotlib.figure.Figure
    """
    children, leaves, depth, ypos, color = _tree_geometry(tree, use_branch_length)
    leafset = set(tree.labels)
    n_leaves = max(len(leaves), 1)
    max_depth = max(depth.values()) or 1.0

    if layout == "rectangular":
        fig, ax = plt.subplots(figsize=(8, max(5, 0.22 * n_leaves)))
        for parent, kids in children.items():
            xp = depth[parent]
            ys = [ypos[c] for c, _ in kids]
            ax.plot(
                [xp, xp],
                [min(ys), max(ys)],
                color=color.get(parent, "#94a3b8"),
                lw=1.1,
            )
            for child, _ in kids:
                ax.plot(
                    [xp, depth[child]],
                    [ypos[child], ypos[child]],
                    color=color.get(child, "#94a3b8"),
                    lw=1.4,
                )
                if child in leafset and show_labels:
                    ax.text(
                        depth[child] + max_depth * 0.01,
                        ypos[child],
                        f" {child}",
                        fontsize=label_size,
                        va="center",
                        color="#334155",
                    )
        ax.set_xlim(-max_depth * 0.02, max_depth * 1.35)
        ax.axis("off")
    else:  # circular
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 8))

        def angle(node: str) -> float:
            return 2 * np.pi * ypos[node] / n_leaves

        def radius(node: str) -> float:
            return 0.12 + 0.88 * depth[node] / max_depth

        for parent, kids in children.items():
            rp = radius(parent)
            child_angles = [angle(c) for c, _ in kids]
            a0, a1 = min(child_angles), max(child_angles)
            arc = np.linspace(a0, a1, 40)
            ax.plot(
                rp * np.cos(arc),
                rp * np.sin(arc),
                color=color.get(parent, "#94a3b8"),
                lw=1.0,
            )
            for child, _ in kids:
                ac = angle(child)
                ax.plot(
                    [rp * np.cos(ac), radius(child) * np.cos(ac)],
                    [rp * np.sin(ac), radius(child) * np.sin(ac)],
                    color=color.get(child, "#94a3b8"),
                    lw=1.3,
                )
                if child in leafset and show_labels:
                    deg = np.degrees(ac)
                    rot = deg if -90 <= deg <= 90 else deg + 180
                    ha = "left" if -90 <= deg <= 90 else "right"
                    rr = radius(child) + 0.02
                    ax.text(
                        rr * np.cos(ac),
                        rr * np.sin(ac),
                        child,
                        fontsize=label_size,
                        rotation=rot,
                        rotation_mode="anchor",
                        ha=ha,
                        va="center",
                        color="#334155",
                    )
        ax.set_aspect("equal")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.axis("off")

    fig.tight_layout()
    _save(fig, path)
    return fig
