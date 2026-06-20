"""Build a human-readable Markdown report from persisted training artifacts.

Reads ``outputs/metrics.json`` and writes ``reports/evaluation_report.md`` with
before/after tropical metrics and the relative improvement. Figures produced by
:mod:`src.train` are referenced by relative path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def generate_report(
    output_dir: str | Path = "outputs",
    report_path: str | Path = "reports/evaluation_report.md",
) -> str:
    """Render an evaluation report from ``metrics.json``.

    Parameters
    ----------
    output_dir : str or pathlib.Path, optional
        Directory containing ``metrics.json`` and the ``figures/`` folder.
    report_path : str or pathlib.Path, optional
        Where to write the Markdown report.

    Returns
    -------
    str
        The Markdown content (also written to ``report_path``).

    Raises
    ------
    FileNotFoundError
        If ``metrics.json`` is missing (run training first).
    """
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run `python -m src.train` (or `python "
            "main.py`) first to produce metrics."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    before = metrics.get("metrics_before", {})
    after = metrics.get("metrics_after", {})
    keys = [
        "mean_violation",
        "median_violation",
        "max_violation",
        "l2_loss",
        "percent_exact",
    ]

    lines = [
        "# Evaluation Report",
        "",
        "> Non-diagnostic research output. Reducing tropical four-point "
        "violations improves mathematical compatibility with an additive tree "
        "metric; it does not establish biological correctness.",
        "",
        "## Run summary",
        "",
        f"- Sequences analyzed: **{metrics.get('n_sequences', 'n/a')}**",
        f"- Relative improvement: **{_fmt(metrics.get('relative_improvement'))}**",
        f"- Runtime (s): {_fmt(metrics.get('runtime_seconds'))}",
        "",
        "## Tropical metrics: before vs after correction",
        "",
        "| Metric | Before (D) | After (X) |",
        "| --- | --- | --- |",
    ]
    for key in keys:
        lines.append(f"| {key} | {_fmt(before.get(key))} | {_fmt(after.get(key))} |")

    lines += [
        "",
        "## Figures",
        "",
        "- `figures/loss_curve.png`",
        "- `figures/violation_histograms.png`",
        "- `figures/before_after_summary.png`",
        "",
        "## Tree artifacts",
        "",
        "- `trees/tree_after.newick`",
        "- `trees/tree_after_edges.csv`",
        "- `trees/tree_after.dot`",
        "",
    ]
    content = "\n".join(lines)

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return content


if __name__ == "__main__":  # pragma: no cover
    print(generate_report())
