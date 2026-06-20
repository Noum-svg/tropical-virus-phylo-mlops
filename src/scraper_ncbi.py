"""OPTIONAL real-data acquisition from NCBI via Biopython Entrez.

This is the *only* way the project obtains sequences it did not already have, and
it acquires **real** records from NCBI -- it never invents sequences. On any
failure (no email configured, network error, empty result) it raises, and the
CLI wrapper exits non-zero.

Configure an Entrez contact email (required by NCBI) in ``params.yaml`` under
``scraper.email`` or the ``NCBI_EMAIL`` environment variable, and optionally an
API key via the environment variable named by ``scraper.api_key_env``.

Run with::

    python -m src.scraper_ncbi --query "Influenza A virus[Organism] AND complete genome" --max-records 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from src.utils import load_params

Record = tuple[str, str]  # (virus_name, rna_sequence)


def fetch_sequences(
    query: str,
    max_records: int,
    email: str,
    api_key: str | None = None,
    database: str = "nucleotide",
    sleep_seconds: float = 0.4,
    max_retries: int = 3,
    batch_size: int = 50,
) -> list[Record]:
    """Fetch real sequences from NCBI Entrez (rate-limited, with retries).

    Parameters
    ----------
    query : str
        Entrez search query.
    max_records : int
        Maximum number of records to retrieve.
    email : str
        Contact email required by NCBI.
    api_key : str or None, optional
        NCBI API key (raises the rate limit).
    database : str, optional
        Entrez database (default ``"nucleotide"``).
    sleep_seconds : float, optional
        Courtesy delay between requests.
    max_retries : int, optional
        Retry attempts per request.
    batch_size : int, optional
        Records fetched per efetch call.

    Returns
    -------
    list of (str, str)
        ``(virus_name, sequence)`` pairs as returned by NCBI.

    Raises
    ------
    ValueError
        If ``email`` is empty.
    RuntimeError
        If the search or fetch ultimately fails.
    """
    if not email:
        raise ValueError(
            "NCBI requires a contact email. Set scraper.email in params.yaml or "
            "the NCBI_EMAIL environment variable."
        )

    from Bio import Entrez, SeqIO  # lazy import; only needed for live acquisition

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    def _with_retry(fn: Callable, what: str):
        last: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - retry any transient failure
                last = exc
                print(f"[scraper] {what} attempt {attempt}/{max_retries} failed: {exc}")
                time.sleep(sleep_seconds * attempt)
        raise RuntimeError(f"{what} failed after {max_retries} attempts: {last}")

    def _search():
        handle = Entrez.esearch(db=database, term=query, retmax=max_records)
        try:
            return Entrez.read(handle)
        finally:
            handle.close()

    search = _with_retry(_search, "esearch")
    id_list = list(search.get("IdList", []))
    if not id_list:
        raise RuntimeError(f"NCBI returned no IDs for query: {query!r}")

    records: list[Record] = []
    for start in range(0, len(id_list), batch_size):
        batch = id_list[start : start + batch_size]
        time.sleep(sleep_seconds)

        def _fetch(ids=batch):
            handle = Entrez.efetch(
                db=database, id=",".join(ids), rettype="fasta", retmode="text"
            )
            try:
                return list(SeqIO.parse(handle, "fasta"))
            finally:
                handle.close()

        for seq_record in _with_retry(_fetch, "efetch"):
            name = (seq_record.description or seq_record.id).strip()
            records.append((name, str(seq_record.seq)))

    if not records:
        raise RuntimeError("NCBI search matched IDs but no sequences were parsed.")
    return records


def write_dataset(records: list[Record], csv_path: str | Path) -> Path:
    """Write fetched records to a CSV; raise if there is nothing to write.

    Parameters
    ----------
    records : list of (str, str)
        ``(virus_name, rna_sequence)`` pairs.
    csv_path : str or pathlib.Path
        Destination path.

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    ValueError
        If ``records`` is empty (sequences are never fabricated to fill a file).
    """
    if not records:
        raise ValueError("Refusing to write an empty dataset; no sequences fetched.")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=["virus_name", "rna_sequence"]).to_csv(
        csv_path, index=False
    )
    return csv_path


def write_stats(
    records: list[Record],
    query: str,
    stats_path: str | Path = "reports/scraper_stats.md",
) -> Path:
    """Write a small provenance/stats Markdown file for an acquisition run."""
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    lengths = [len(seq) for _, seq in records]
    n = len(records)
    mean_len = sum(lengths) / n if n else 0
    lines = [
        "# NCBI Acquisition Stats",
        "",
        f"- Query: `{query}`",
        f"- Records fetched: **{n}**",
        f"- Min length: {min(lengths) if lengths else 0}",
        f"- Max length: {max(lengths) if lengths else 0}",
        f"- Mean length: {mean_len:.1f}",
        "",
        "Source: NCBI Entrez. These are real records; provenance, licensing, and "
        "privacy controls are the user's responsibility.",
        "",
    ]
    stats_path.write_text("\n".join(lines), encoding="utf-8")
    return stats_path


def run_scraper(
    query: str,
    max_records: int = 100,
    params: dict | None = None,
    output_path: str | Path | None = None,
    stats_path: str | Path = "reports/scraper_stats.md",
    fetcher: Callable[..., list[Record]] = fetch_sequences,
) -> Path:
    """Acquire real sequences and persist the dataset and stats.

    Parameters
    ----------
    query : str
        Entrez search query.
    max_records : int, optional
        Maximum records to fetch.
    params : dict or None, optional
        Configuration; loaded from ``params.yaml`` when ``None``.
    output_path : str or pathlib.Path or None, optional
        CSV destination (defaults to ``params['data']['raw_path']``).
    fetcher : callable, optional
        Sequence fetcher (injectable for testing). Defaults to
        :func:`fetch_sequences`.

    Returns
    -------
    pathlib.Path
        Path to the written CSV.

    Raises
    ------
    ValueError
        If no contact email is configured or the result is empty.
    """
    params = params or load_params()
    scraper_cfg = params.get("scraper", {})
    email = scraper_cfg.get("email") or os.environ.get("NCBI_EMAIL")
    api_key = os.environ.get(scraper_cfg.get("api_key_env", "NCBI_API_KEY"))
    output_path = output_path or params["data"]["raw_path"]

    records = fetcher(
        query=query,
        max_records=max_records,
        email=email,
        api_key=api_key,
        database=scraper_cfg.get("database", "nucleotide"),
        sleep_seconds=scraper_cfg.get("sleep_seconds", 0.4),
        max_retries=scraper_cfg.get("max_retries", 3),
    )
    csv_path = write_dataset(records, output_path)
    write_stats(records, query, stats_path)
    print(f"[scraper] Wrote {len(records)} real sequences to {csv_path}")
    return csv_path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(
        description="Acquire real viral sequences from NCBI."
    )
    parser.add_argument("--query", required=True, help="Entrez search query.")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--output", default=None, help="Output CSV path.")
    args = parser.parse_args(argv)
    try:
        run_scraper(args.query, max_records=args.max_records, output_path=args.output)
    except Exception as exc:  # noqa: BLE001 - surface any failure as non-zero exit
        print(f"[scraper] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
