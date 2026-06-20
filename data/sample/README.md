# Synthetic demonstration data

⚠️ **`sample_viral_sequences.csv` in this folder is SYNTHETIC demonstration data.**

It was produced by `scripts/generate_demo_dataset.py` (fixed seed) purely so the
dashboard and CI have something runnable without network access or private data.

- It is **not** real viral data.
- It must **never** be presented as a scientific result.
- The scientific pipeline never fabricates sequences. For real analyses, place a
  curated CSV at `data/raw/multi_virus_rna_sequences_100.csv` (columns
  `virus_name,rna_sequence`) or acquire data with `python -m src.scraper_ncbi`.
