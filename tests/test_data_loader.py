"""Tests for :mod:`src.data_loader` -- cleaning, schema validation, filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    CLEAN_COLUMNS,
    clean_dataframe,
    clean_sequence,
    load_and_clean,
    load_raw_csv,
)


def test_clean_uppercases_and_converts_u_to_t():
    assert clean_sequence("augc") == "ATGC"
    assert clean_sequence("AuGcUu") == "ATGCTT"
    assert clean_sequence("uuuu") == "TTTT"


def test_clean_removes_invalid_characters():
    assert clean_sequence("ACGTN-XYZ123") == "ACGT"
    assert clean_sequence("acgt rn") == "ACGT"  # 'r','n' and space dropped


def test_clean_removes_whitespace():
    assert clean_sequence("AC GT\n\tAC") == "ACGTAC"


def test_clean_empty_and_nan():
    assert clean_sequence("") == ""
    assert clean_sequence("NNN---") == ""
    assert clean_sequence(None) == ""
    assert clean_sequence(float("nan")) == ""


def test_load_raw_csv_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_csv(tmp_path / "does_not_exist.csv")


def test_load_raw_csv_bad_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["a"], "seq": ["ACGT"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_raw_csv(path)


def test_clean_dataframe_filters_short_empty_and_duplicates():
    raw = pd.DataFrame(
        {
            "virus_name": ["A", "B", "A", "C", ""],
            "rna_sequence": ["AUGCAUGC", "AC", "AUGCAUGC", "NNNN", "ACGTACGT"],
        }
    )
    out = clean_dataframe(raw, min_seq_length=5)

    # B too short (len 2); C empty after cleaning; the empty-name row dropped;
    # the duplicate (A, AUGCAUGC) collapsed -> only one A row survives.
    assert list(out["virus_name"]) == ["A"]
    assert list(out["clean_sequence"]) == ["ATGCATGC"]
    assert list(out["sequence_length"]) == [8]
    assert tuple(out.columns) == CLEAN_COLUMNS


def test_clean_dataframe_rejects_bad_min_length():
    raw = pd.DataFrame({"virus_name": ["A"], "rna_sequence": ["ACGT"]})
    with pytest.raises(ValueError):
        clean_dataframe(raw, min_seq_length=0)


def test_clean_dataframe_missing_columns_raises():
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame({"x": [1]}))


def test_load_and_clean_roundtrip(tmp_path):
    path = tmp_path / "ok.csv"
    pd.DataFrame(
        {
            "virus_name": ["V1", "V2"],
            "rna_sequence": ["AUGGUGAAGU", "ACGTACGTAC"],
        }
    ).to_csv(path, index=False)

    out = load_and_clean(path, min_seq_length=5)
    assert len(out) == 2
    assert out["clean_sequence"].iloc[0] == "ATGGTGAAGT"
    assert out["sequence_length"].iloc[0] == 10
