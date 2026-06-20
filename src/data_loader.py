"""Load, validate, and clean viral sequence CSVs.

Input contract (exact column names)::

    virus_name,rna_sequence

This module never fabricates sequences. If the configured CSV is missing,
:func:`load_raw_csv` raises a clear, actionable error instructing the user to
provide real data or run the NCBI scraper.

Cleaning (applied to every ``rna_sequence``):

1. uppercase;
2. remove all whitespace;
3. replace ``U`` with ``T``;
4. keep only the characters ``A``, ``C``, ``G``, ``T``.

The cleaned DataFrame is then filtered: empty identifiers are dropped,
sequences shorter than ``min_seq_length`` are rejected, and duplicate
``(virus_name, clean_sequence)`` rows are removed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

#: Exact required input columns.
REQUIRED_COLUMNS: tuple[str, ...] = ("virus_name", "rna_sequence")

#: Output columns of the cleaned DataFrame, in order.
CLEAN_COLUMNS: tuple[str, ...] = (
    "virus_name",
    "rna_sequence",
    "clean_sequence",
    "sequence_length",
)

#: Nucleotides retained after cleaning.
VALID_BASES: frozenset[str] = frozenset("ACGT")

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ACGT_RE = re.compile(r"[^ACGT]")


def clean_sequence(seq: object) -> str:
    """Clean a single raw sequence into an ``ACGT``-only string.

    Steps, in order: uppercase, remove whitespace, replace ``U`` with ``T``,
    drop any character that is not ``A``, ``C``, ``G``, or ``T``. The result
    may be the empty string.

    Parameters
    ----------
    seq : object
        Raw sequence value (typically ``str``; ``NaN``/``None`` becomes ``""``).

    Returns
    -------
    str
        The cleaned nucleotide string.
    """
    if seq is None or (isinstance(seq, float) and pd.isna(seq)):
        return ""
    s = str(seq).upper()
    s = _WHITESPACE_RE.sub("", s)
    s = s.replace("U", "T")
    return _NON_ACGT_RE.sub("", s)


def _check_columns(df: pd.DataFrame, source: str) -> None:
    """Raise ``ValueError`` if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s): {missing}. "
            f"Required columns are exactly {list(REQUIRED_COLUMNS)}."
        )


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and validate its schema.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input CSV.

    Returns
    -------
    pandas.DataFrame
        The raw DataFrame (uncleaned), guaranteed to contain the required
        columns.

    Raises
    ------
    FileNotFoundError
        If the file does not exist. The message explains how to obtain real
        data; synthetic sequences are never generated.
    ValueError
        If the required columns are absent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Input CSV not found at '{p}'.\n"
            "Place a real viral-sequence CSV there with columns "
            "'virus_name,rna_sequence', or run `python -m src.scraper_ncbi` to "
            "fetch sequences from NCBI.\n"
            "This is a research tool: it never fabricates viral sequences."
        )
    df = pd.read_csv(p)
    _check_columns(df, f"CSV at '{p}'")
    return df


def clean_dataframe(df: pd.DataFrame, min_seq_length: int = 1) -> pd.DataFrame:
    """Clean and filter a raw DataFrame into the canonical cleaned form.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data containing at least the required columns.
    min_seq_length : int, optional
        Minimum length of a cleaned sequence to keep. Sequences shorter than
        this (including empty ones) are dropped. Defaults to ``1``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns :data:`CLEAN_COLUMNS` and a reset index. Empty
        identifiers are dropped, short sequences rejected, and duplicate
        ``(virus_name, clean_sequence)`` rows removed.

    Raises
    ------
    ValueError
        If the required columns are absent or ``min_seq_length < 1``.
    """
    _check_columns(df, "DataFrame")
    if min_seq_length < 1:
        raise ValueError(f"min_seq_length must be >= 1, got {min_seq_length}.")

    out = df.copy()
    out["virus_name"] = out["virus_name"].astype(str).str.strip()
    out["clean_sequence"] = out["rna_sequence"].apply(clean_sequence)
    out["sequence_length"] = out["clean_sequence"].str.len()

    keep = (out["virus_name"].str.len() > 0) & (
        out["sequence_length"] >= min_seq_length
    )
    out = out.loc[keep]
    out = out.drop_duplicates(subset=["virus_name", "clean_sequence"])
    out = out.reset_index(drop=True)
    return out.loc[:, list(CLEAN_COLUMNS)]


def load_and_clean(path: str | Path, min_seq_length: int = 1) -> pd.DataFrame:
    """Load the raw CSV and return the cleaned, filtered DataFrame.

    Convenience wrapper around :func:`load_raw_csv` and :func:`clean_dataframe`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input CSV.
    min_seq_length : int, optional
        Minimum cleaned sequence length to keep. Defaults to ``1``.

    Returns
    -------
    pandas.DataFrame
        The cleaned DataFrame with columns :data:`CLEAN_COLUMNS`.
    """
    return clean_dataframe(load_raw_csv(path), min_seq_length=min_seq_length)


def sequence_summary(clean_df: pd.DataFrame) -> dict[str, object]:
    """Summary statistics for a cleaned sequence DataFrame (for dashboards).

    Parameters
    ----------
    clean_df : pandas.DataFrame
        Output of :func:`clean_dataframe`.

    Returns
    -------
    dict
        ``n_sequences``, ``min_length``, ``max_length``, ``mean_length`` and
        ``gc_content`` (fraction of G/C across all cleaned sequences).
    """
    n = len(clean_df)
    lengths = clean_df["sequence_length"]
    joined = "".join(clean_df["clean_sequence"]) if n else ""
    gc = (joined.count("G") + joined.count("C")) / len(joined) if joined else 0.0
    return {
        "n_sequences": int(n),
        "min_length": int(lengths.min()) if n else 0,
        "max_length": int(lengths.max()) if n else 0,
        "mean_length": float(lengths.mean()) if n else 0.0,
        "gc_content": float(gc),
    }


if __name__ == "__main__":  # pragma: no cover - DVC `load_clean` stage
    from src.utils import load_params

    params = load_params()
    cleaned = load_and_clean(
        params["data"]["raw_path"], min_seq_length=params["data"]["min_seq_length"]
    )
    out = Path(params["data"]["processed_dir"]) / "clean_sequences.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out, index=False)
    print(f"[data_loader] wrote {len(cleaned)} cleaned sequences to {out}")
