r"""Neighbor-Joining reconstruction and tree serialization.

Given the corrected matrix ``X`` (or any distance matrix ``M``), Neighbor-Joining
repeatedly joins the closest pair under the ``Q`` criterion::

    r_i  = sum_k M[i, k]                       (active row sum)
    Q_ij = (m - 2) * M[i, j] - r_i - r_j       (m = active node count)

The minimizing pair ``(a, b)`` is merged into a new internal node ``u`` with::

    l_a   = 0.5 * M[a, b] + (r_a - r_b) / (2 * (m - 2))
    l_b   = M[a, b] - l_a
    M[u,k]= 0.5 * (M[a, k] + M[b, k] - M[a, b])  for each remaining k

When two nodes remain they are joined by an edge of length ``M[a, b]``. Negative
branch lengths are clamped to ``0`` for display. The same internal
:class:`PhyloTree` is exported as a **Newick** string, an **edge list**
(``parent, child, branch_length``), and a **DOT/Graphviz** string.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_NEWICK_SPECIAL = re.compile(r"[ \t\n(),:;\[\]']")


def _escape_newick(name: str) -> str:
    """Quote a label for Newick if it contains reserved characters."""
    s = str(name)
    if _NEWICK_SPECIAL.search(s):
        return "'" + s.replace("'", "''") + "'"
    return s


def _fmt(length: float) -> str:
    """Format a branch length for Newick/DOT output."""
    return f"{float(length):.6f}"


@dataclass
class PhyloTree:
    """A reconstructed tree as a directed edge list rooted at :attr:`root`.

    Attributes
    ----------
    edges : list of dict
        Each edge is ``{"parent": str, "child": str, "branch_length": float}``
        with branch lengths already clamped to ``>= 0``.
    root : str
        Identifier of the root node (one of the last two active nodes).
    labels : list of str
        Original leaf labels (taxon names).
    """

    edges: list[dict] = field(default_factory=list)
    root: str = ""
    labels: list[str] = field(default_factory=list)

    def _children_map(self) -> dict[str, list[tuple[str, float]]]:
        cmap: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for e in self.edges:
            cmap[e["parent"]].append((e["child"], e["branch_length"]))
        return cmap

    def to_newick(self) -> str:
        """Serialize to a Newick string (internal node names are omitted)."""
        cmap = self._children_map()
        leafset = set(self.labels)

        def rec(node: str) -> str:
            kids = cmap.get(node, [])
            if not kids:  # genuine leaf
                return _escape_newick(node)
            inner = ",".join(f"{rec(c)}:{_fmt(bl)}" for c, bl in kids)
            name = _escape_newick(node) if node in leafset else ""
            return f"({inner}){name}"

        if not self.edges and not self.labels:
            return ";"
        if not self.edges:  # single leaf
            return f"{_escape_newick(self.labels[0])};"
        return f"{rec(self.root)};"

    def to_edge_list(self) -> list[dict]:
        """Return the edge list as a list of JSON-serializable dictionaries."""
        return [dict(e) for e in self.edges]

    def to_edge_dataframe(self) -> pd.DataFrame:
        """Return the edge list as a DataFrame (``parent, child, branch_length``)."""
        return pd.DataFrame(self.edges, columns=["parent", "child", "branch_length"])

    def to_dot(self) -> str:
        """Serialize to a DOT/Graphviz digraph string."""
        leafset = set(self.labels)
        lines = ["digraph PhyloTree {", "  rankdir=LR;", "  node [fontsize=10];"]
        nodes = {self.root}
        for e in self.edges:
            nodes.add(e["parent"])
            nodes.add(e["child"])
        for node in sorted(nodes):
            shape = "box" if node in leafset else "ellipse"
            lines.append(f'  "{node}" [shape={shape}];')
        for e in self.edges:
            lines.append(
                f'  "{e["parent"]}" -> "{e["child"]}" '
                f'[label="{_fmt(e["branch_length"])}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    def summary(self) -> dict[str, float]:
        """Return tree summary stats (leaves, internal nodes, total branch length)."""
        leafset = set(self.labels)
        nodes = {e["parent"] for e in self.edges} | {e["child"] for e in self.edges}
        internal = {n for n in nodes if n not in leafset}
        total_bl = sum(float(e["branch_length"]) for e in self.edges)
        return {
            "n_leaves": len(self.labels),
            "n_internal": len(internal),
            "total_branch_length": float(total_bl),
        }


def neighbor_joining(matrix: np.ndarray, labels: list[str]) -> PhyloTree:
    """Reconstruct a tree from a distance matrix via Neighbor-Joining.

    Parameters
    ----------
    matrix : numpy.ndarray
        Symmetric ``n x n`` distance matrix.
    labels : list of str
        ``n`` leaf labels, aligned with the matrix rows.

    Returns
    -------
    PhyloTree
        The reconstructed tree. For ``n == 1`` the tree is a single labeled leaf
        with no edges.

    Raises
    ------
    ValueError
        If the matrix is not square or ``labels`` length mismatches.
    """
    m_full = np.asarray(matrix, dtype=float)
    if m_full.ndim != 2 or m_full.shape[0] != m_full.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {m_full.shape}.")
    n = m_full.shape[0]
    if len(labels) != n:
        raise ValueError(f"labels has length {len(labels)} but matrix is {n}x{n}.")

    labels = [str(x) for x in labels]
    leafset = set(labels)
    if n == 0:
        return PhyloTree(edges=[], root="", labels=[])
    if n == 1:
        return PhyloTree(edges=[], root=labels[0], labels=labels)
    if n == 2:
        # No NJ step; attach both leaves to a synthetic internal root so each
        # leaf stays a terminal. Split the distance to preserve the path length.
        half = max(0.0, float(m_full[0, 1])) / 2.0
        edges = [
            {"parent": "node_1", "child": labels[0], "branch_length": half},
            {"parent": "node_1", "child": labels[1], "branch_length": half},
        ]
        return PhyloTree(edges=edges, root="node_1", labels=labels)

    matrix = m_full.copy()
    names = list(labels)
    edges: list[dict] = []
    internal_count = 0

    while len(names) > 2:
        m = len(names)
        r = matrix.sum(axis=1)
        q = (m - 2) * matrix - r[:, None] - r[None, :]
        np.fill_diagonal(q, np.inf)

        a, b = divmod(int(np.argmin(q)), m)
        if a > b:
            a, b = b, a

        d_ab = matrix[a, b]
        l_a = 0.5 * d_ab + (r[a] - r[b]) / (2 * (m - 2))
        l_b = d_ab - l_a

        internal_count += 1
        u = f"node_{internal_count}"
        edges.append(
            {"parent": u, "child": names[a], "branch_length": max(0.0, float(l_a))}
        )
        edges.append(
            {"parent": u, "child": names[b], "branch_length": max(0.0, float(l_b))}
        )

        keep = [i for i in range(m) if i not in (a, b)]
        u_dist = 0.5 * (matrix[a, keep] + matrix[b, keep] - d_ab)

        size = len(keep)
        new_matrix = np.zeros((size + 1, size + 1))
        new_matrix[:size, :size] = matrix[np.ix_(keep, keep)]
        new_matrix[:size, size] = u_dist
        new_matrix[size, :size] = u_dist

        names = [names[i] for i in keep] + [u]
        matrix = new_matrix

    # Two active nodes remain: join them. Root at the internal node when exactly
    # one of the pair is a leaf, so every leaf stays a terminal in the tree.
    final_length = max(0.0, float(matrix[0, 1]))
    x, y = names[0], names[1]
    if x in leafset and y not in leafset:
        root, child = y, x
    else:
        root, child = x, y
    edges.append({"parent": root, "child": child, "branch_length": final_length})
    return PhyloTree(edges=edges, root=root, labels=labels)


def reconstruct_tree(
    matrix: np.ndarray | pd.DataFrame, labels: list[str] | None = None
) -> PhyloTree:
    """Reconstruct a tree, taking labels from a DataFrame index when available.

    Parameters
    ----------
    matrix : numpy.ndarray or pandas.DataFrame
        Distance matrix. If a DataFrame and ``labels`` is ``None``, the index is
        used for labels.
    labels : list of str or None, optional
        Explicit labels; required when ``matrix`` is a bare array.

    Returns
    -------
    PhyloTree
        The reconstructed tree.
    """
    if isinstance(matrix, pd.DataFrame):
        if labels is None:
            labels = [str(x) for x in matrix.index]
        values = matrix.to_numpy(dtype=float)
    else:
        values = np.asarray(matrix, dtype=float)
        if labels is None:
            labels = [f"seq_{i}" for i in range(values.shape[0])]
    return neighbor_joining(values, labels)


if __name__ == "__main__":  # pragma: no cover - DVC `phylogenetic_tree` stage
    from pathlib import Path

    from src.utils import load_params

    params = load_params()
    out_dir = Path(params["outputs"]["output_dir"])
    tree_dir = Path(params["outputs"]["trees_dir"])
    corrected = pd.read_csv(out_dir / "corrected_distance_matrix.csv", index_col=0)
    corrected.columns = corrected.index
    built = reconstruct_tree(corrected)
    tree_dir.mkdir(parents=True, exist_ok=True)
    (tree_dir / "tree_after.newick").write_text(built.to_newick(), encoding="utf-8")
    built.to_edge_dataframe().to_csv(tree_dir / "tree_after_edges.csv", index=False)
    (tree_dir / "tree_after.dot").write_text(built.to_dot(), encoding="utf-8")
    print(f"[phylogeny] wrote tree artifacts to {tree_dir}")
