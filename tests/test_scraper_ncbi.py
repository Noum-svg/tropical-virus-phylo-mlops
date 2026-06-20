"""Tests for :mod:`src.scraper_ncbi` -- mocked NCBI; no live network access."""

from __future__ import annotations

import pandas as pd
import pytest

from src.scraper_ncbi import fetch_sequences, run_scraper, write_dataset


def _fake_fetcher(**kwargs):
    # Toy records standing in for an NCBI response (test input, not a result).
    return [
        ("Virus alpha complete genome", "ACGTACGTACGT"),
        ("Virus beta complete genome", "ACGTACGAACGT"),
        ("Virus gamma complete genome", "ACGAACGTACGT"),
    ]


def test_run_scraper_writes_csv_and_stats(tmp_path):
    csv_path = tmp_path / "raw.csv"
    stats_path = tmp_path / "scraper_stats.md"
    returned = run_scraper(
        query="dummy[Organism]",
        max_records=10,
        params={"data": {"raw_path": str(csv_path)}, "scraper": {}},
        output_path=csv_path,
        stats_path=stats_path,
        fetcher=_fake_fetcher,
    )
    assert returned == csv_path
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["virus_name", "rna_sequence"]
    assert len(df) == 3
    assert stats_path.exists()
    assert "Records fetched" in stats_path.read_text(encoding="utf-8")


def test_write_dataset_refuses_empty(tmp_path):
    with pytest.raises(ValueError):
        write_dataset([], tmp_path / "empty.csv")


def test_run_scraper_propagates_empty_failure(tmp_path):
    with pytest.raises(ValueError):
        run_scraper(
            query="dummy",
            params={"data": {"raw_path": str(tmp_path / "x.csv")}, "scraper": {}},
            output_path=tmp_path / "x.csv",
            stats_path=tmp_path / "s.md",
            fetcher=lambda **kw: [],  # NCBI returned nothing -> must raise, never fake
        )


def test_fetch_sequences_requires_email():
    with pytest.raises(ValueError):
        fetch_sequences(query="q", max_records=1, email="")
