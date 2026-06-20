"""CLI entry point: run the full Tropical Virus PhyloTree pipeline end-to-end.

Examples
--------
Real data (must exist at the configured path, never fabricated)::

    python main.py

Bundled SYNTHETIC demonstration data (for a quick local run)::

    python main.py --demo
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from src.evaluate import generate_report
from src.train import run_training
from src.utils import load_params

DEMO_CSV = "data/sample/sample_viral_sequences.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Tropical Virus PhyloTree pipeline (D -> X -> tree)."
    )
    parser.add_argument(
        "--input", default=None, help="Input CSV (default: params data.raw_path)."
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the bundled SYNTHETIC demo dataset (not real viral data).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow logging."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    params = load_params(args.params)

    input_path = args.input
    if args.demo:
        input_path = DEMO_CSV
        # Demo sequences are short; relax the length filter so they survive.
        params["data"]["min_seq_length"] = 1
        print("[main] Using SYNTHETIC demo data -- not a scientific result.")

    summary = run_training(
        params=params, input_path=input_path, enable_mlflow=not args.no_mlflow
    )
    generate_report(output_dir=params["outputs"]["output_dir"])
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
