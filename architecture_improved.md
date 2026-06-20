# System Architecture (v2 summary)

> Reconciled to the implemented build. The full reference is
> [`docs/architecture.md`](docs/architecture.md); this page summarizes the actual
> module layout.

One scientific implementation in `src/`; `api/`, `app/`, and the CLI only
orchestrate it (they never reimplement distance, four-point, optimization, or
Neighbor-Joining math). Dependencies point from delivery surfaces toward the
core; `src/` never imports `api/` or `app/`.

```
data_loader → distances → tropical_grassmannian → tropical_gradient_descent → phylogeny → predict
utils (shared leaf: indexing, projection, validation, quadruplets, load_params)
visualization (matplotlib, Agg)   scraper_ncbi (optional NCBI Entrez acquisition)
```

## Implemented modules

| Module | Responsibility |
| --- | --- |
| `src/utils.py` | pair↔index, matrix↔vector, `secondmax`, `tropical_norm`, quadruplets, `project_to_distance_space`, `is_valid_distance_matrix`, `load_params` |
| `src/data_loader.py` | CSV load, schema validation, ACGT cleaning, filtering |
| `src/distances.py` | pairwise distance, matrix `D` |
| `src/tropical_grassmannian.py` | four-point sums, violations, `tropical_score` |
| `src/tropical_gradient_descent.py` | loss, subgradient, `fit`, `correct_distance_matrix` |
| `src/phylogeny.py` | Neighbor-Joining, `PhyloTree`, Newick/edges/DOT |
| `src/predict.py` | sequences → D → X → tree orchestration |
| `src/train.py`, `src/evaluate.py` | batch run + artifacts + MLflow; Markdown report |
| `src/visualization.py` | loss curve, violation histograms, summary figures |
| `src/scraper_ncbi.py` | rate-limited NCBI Entrez acquisition (never fabricates) |
| `api/main.py` | FastAPI: 7 endpoints, pydantic, JSON boundary |
| `app/streamlit_app.py`, `app/ui_helpers.py` | 10-page dashboard, figures |

> Note: the actual acquisition module is `scraper_ncbi.py` (the older draft
> called it `data_acquisition.py`), and there is no separate `components.py`;
> reusable UI lives in `ui_helpers.py`.

## Invariants and reproducibility

`utils.project_to_distance_space` enforces $M=M^\top$, $M_{ii}=0$, $M_{ij}\ge0$;
`utils.is_valid_distance_matrix` checks them at boundaries. `params.yaml` holds all
configuration; every stochastic step reads `seed`. `dvc.yaml` rebuilds end-to-end:
`load_clean → distance_matrix → tropical_correction → phylogenetic_tree`. The
committed `data/sample/sample_viral_sequences.csv` is **synthetic** demo data
(dashboard/CI only), never presented as a real result.
