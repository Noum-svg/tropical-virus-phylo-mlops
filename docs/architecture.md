# Architecture

One scientific implementation in `src/`; the API, dashboard, and CLI only
orchestrate it. Dependencies point from delivery surfaces toward the core —
`src/` never imports `api/` or `app/`.

```
data_loader → distances → tropical_grassmannian → tropical_gradient_descent → phylogeny → predict
                                                                                              │
                                              train / evaluate / api / app  ─────────────────┘
utils ......... shared leaf helpers (indexing, projection, validation, quadruplets)
visualization . matplotlib figures (Agg)
scraper_ncbi .. optional real-data acquisition (writes a CSV the loader reads)
```

## Module responsibilities

| Module | Responsibility | Key outputs |
| --- | --- | --- |
| `src/utils.py` | pair↔index, matrix↔vector, `secondmax`, `tropical_norm`, seeded quadruplets, projection, validation, `load_params` | helpers |
| `src/data_loader.py` | CSV load, schema validation, cleaning, filtering | clean DataFrame |
| `src/distances.py` | pairwise distance, matrix `D` | `D` (DataFrame) |
| `src/tropical_grassmannian.py` | four-point sums, vectorized violations, `tropical_score` | metrics |
| `src/tropical_gradient_descent.py` | loss, subgradient, `fit`, `correct_distance_matrix` | `omega`, `X`, history |
| `src/phylogeny.py` | Neighbor-Joining, `PhyloTree`, Newick/edges/DOT | tree |
| `src/predict.py` | sequences → D → X → tree orchestration | result dict |
| `src/train.py` | batch run, artifacts, MLflow logging | `outputs/`, `models/omega.npy` |
| `src/evaluate.py` | Markdown report from `metrics.json` | `reports/evaluation_report.md` |
| `src/visualization.py` | loss curve, violation histograms, summary | PNGs |
| `src/scraper_ncbi.py` | rate-limited NCBI Entrez acquisition | raw CSV, stats |
| `api/main.py` | FastAPI: 7 endpoints, pydantic, JSON boundary | JSON |
| `app/streamlit_app.py`, `app/ui_helpers.py` | 10-page dashboard, figures | UI |

## Matrix invariants

Every distance-like matrix satisfies $M=M^\top$, $M_{ii}=0$, $M_{ij}\ge0$, all
finite. `utils.project_to_distance_space` enforces them; `utils.is_valid_distance_matrix`
checks them at module boundaries.

## Reproducibility

`params.yaml` holds all configuration (loaded once, passed down). Every
stochastic step (quadruplet sampling) reads the `seed`. The DVC pipeline
(`dvc.yaml`) rebuilds end-to-end: `load_clean → distance_matrix →
tropical_correction → phylogenetic_tree`.

## Data policy

Real sequences are never fabricated. A missing input CSV raises a clear error.
`data/sample/sample_viral_sequences.csv` is **synthetic** demonstration data for
the dashboard and CI only, clearly labeled and never presented as a result.

## Container architecture

Docker Compose runs one responsibility per container on the shared
`tropical-virus-phylo-network` network:

```text
browser
  |
  | http://localhost:8080
  v
frontend (nginx + React)
  |
  | reverse proxy via service DNS
  v
backend:8000 (FastAPI)
  ^
  | API_BASE=http://backend:8000
  |
streamlit:8501
```

The images are defined independently:

| Service | Dockerfile | Runtime |
| --- | --- | --- |
| `backend` | `docker/Dockerfile.backend` | Python 3.12 + FastAPI |
| `frontend` | `docker/Dockerfile.frontend` | Node 22 build + nginx |
| `streamlit` | `docker/Dockerfile.streamlit` | Python 3.12 + Streamlit |

nginx serves the SPA with `try_files $uri /index.html` and proxies all public
API paths to `http://backend:8000`. Persistent host mounts keep `data/`,
`outputs/`, and `models/` outside the containers.
