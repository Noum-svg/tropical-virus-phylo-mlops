"""Generate CLEARLY-SYNTHETIC demonstration data for UI/CI smoke tests ONLY.

WARNING
-------
The CSV produced by this script is **synthetic demonstration data**. It is NOT
real viral data and must never be presented as a scientific result. The
scientific pipeline (``src/``) never fabricates sequences; real data must be a
curated CSV in ``data/raw/`` or acquired with ``src/scraper_ncbi.py``. This
offline helper exists only so the dashboard and CI have something runnable.

The data is generated deterministically (fixed seed): a few "cluster ancestors"
are each mutated into several descendants, yielding a non-trivial tree shape.
Run with:  ``python scripts/generate_demo_dataset.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT_PATH = Path("data/sample/sample_viral_sequences.csv")
BASES = np.array(list("ACGU"))  # RNA alphabet; the loader maps U -> T
SEED = 42
N_CLUSTERS = 3
PER_CLUSTER = 4
SEQ_LENGTH = 60
MUTATION_RATE = 0.10


def _random_sequence(rng: np.random.Generator, length: int) -> list[str]:
    return list(rng.choice(BASES, size=length))


def _mutate(rng: np.random.Generator, seq: list[str], rate: float) -> str:
    out = list(seq)
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] = rng.choice(BASES)
    return "".join(out)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: list[tuple[str, str]] = []
    for c in range(N_CLUSTERS):
        ancestor = _random_sequence(rng, SEQ_LENGTH)
        for k in range(PER_CLUSTER):
            name = f"SYNTHETIC_DEMO_c{c + 1}_t{k + 1}"
            rows.append((name, _mutate(rng, ancestor, MUTATION_RATE)))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as fh:
        fh.write("virus_name,rna_sequence\n")
        for name, seq in rows:
            fh.write(f"{name},{seq}\n")
    print(f"Wrote {len(rows)} SYNTHETIC demo sequences to {OUT_PATH}")


if __name__ == "__main__":
    main()
