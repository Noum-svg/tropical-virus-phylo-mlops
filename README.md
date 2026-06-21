# 🧬 Tropical Virus PhyloTree MLOps

[![CI](https://github.com/Noum-svg/tropical-virus-phylo-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Noum-svg/tropical-virus-phylo-mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

Transform viral RNA/DNA sequences (CSV) into an **optimized phylogenetic tree** by
learning a *tropical correction* of a sequence distance matrix and then running
Neighbor-Joining. It ships with a professional **React** dashboard, a **FastAPI**
service, a **Streamlit** app, and one-command **Docker** deployment.

![Professional React dashboard](docs/images/dashboard-overview.png)

> ⚠️ **Non-diagnostic research tool.** This project analyzes viral sequence
> relationships and reconstructs trees for computational-biology, research, and
> educational purposes. It is **not** a medical diagnostic system and must not
> be presented as diagnosing infection or disease. Reducing tropical
> four-point violations improves mathematical compatibility with an additive
> tree metric; it does **not** prove the inferred topology is the true
> evolutionary history.

> 🚫 **No synthetic data as results.** Real viral sequences are never
> fabricated. If the input CSV is missing, the pipeline raises a clear error
> telling you to provide real data or run the NCBI scraper. (Small toy
> sequences inside `tests/` are unit-test inputs only.)

## The math

The pipeline learns a symmetric, zero-diagonal correction $\omega$ and forms the
corrected distance matrix

```math
X = D + \omega
```

where $D$ is the initial sequence-distance matrix. $\omega$ is optimized to reduce
the **tropical four-point violations** of $X$; Neighbor-Joining then reconstructs
the tree from $X$. The distance matrix is an internal artifact — the **tree**
(Newick + edge list + DOT) is the deliverable.

### Sequence distance

For cleaned sequences $s_i, s_j$ with lengths $L_i, L_j$ and shared length
$L_{ij} = \min(L_i, L_j)$, the normalized Hamming term $H$ and the length
penalty $P$ combine into the distance $d \in [0, 1]$:

```math
H(s_i, s_j) = \frac{1}{L_{ij}} \sum_{k=1}^{L_{ij}} \mathbf{1}[\, s_i(k) \neq s_j(k) \,],
\qquad
P(s_i, s_j) = \frac{| L_i - L_j |}{\max(L_i, L_j)}
```

```math
d(s_i, s_j) = \alpha \, H(s_i, s_j) + (1 - \alpha)\, P(s_i, s_j), \qquad \alpha = 0.9
```

### Tropical four-point violation

For each quadruplet $i < j < k < l$ of a symmetric matrix $X$, form the three
pairwise sums and take the gap between the largest two:

```math
S_1 = X_{ij} + X_{kl}, \qquad S_2 = X_{ik} + X_{jl}, \qquad S_3 = X_{il} + X_{jk}
```

```math
\delta_{ijkl}(X) = \max(S_1, S_2, S_3) - \mathrm{secondmax}(S_1, S_2, S_3) \; \ge \; 0
```

The quadruplet is **tree-compatible** exactly when $\delta_{ijkl}(X) = 0$ (the
maximum is attained at least twice).

### Objective and optimizer

Let $Q$ denote the evaluated quadruplets and let $g$ be the current
subgradient. Define the tropical gradient spread

```math
T(g) = \max(g) - \min(g).
```

The correction minimizes the total squared violation with an explicit squared
matrix penalty:

```math
L(\omega) = \sum_{q \in Q} \delta_q(D + \omega)^2
            + \lambda \sum_{i,j} \omega_{ij}^2.
```

The optimizer normalizes its subgradient step by $T(g)$:

```math
\omega_{t+1} = \omega_t
               - \frac{\gamma}{T(g_t) + \varepsilon} g_t.
```

The improvement reported after correction is the relative drop in the squared
four-point loss $L_2(M)=\sum_{q\in Q}\delta_q(M)^2$:

```math
\mathrm{RI} = \frac{L_2(D) - L_2(X)}{L_2(D) + \varepsilon}
```

## Project layout

```
src/         scientific core (pure, typed, documented functions)
api/         FastAPI app (orchestration only)
app/         Streamlit dashboard (orchestration only)
tests/       pytest suite
data/raw/    real input CSV (git-ignored; never committed)
outputs/     generated matrices, omega, history, metrics, trees, figures (git-ignored)
params.yaml  central configuration (loaded once, passed down)
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run the tests

```bash
python -m pytest -q          # all green
black --check . && ruff check .
```

## Run the pipeline

The pipeline requires **real** data at `data/raw/multi_virus_rna_sequences_100.csv`
(`virus_name,rna_sequence`). If it is absent, the run raises a clear error —
sequences are never fabricated. Acquire real data from NCBI, or run on the
bundled clearly-labeled **synthetic** demo:

```bash
# Real data (place the CSV first, or fetch it):
python -m src.scraper_ncbi --query "Influenza A virus[Organism] AND complete genome" --max-records 100
python main.py                       # full pipeline end-to-end
#   OR rebuild via DVC stages:
dvc repro

# Quick local run on the synthetic demo (not a scientific result):
python main.py --demo --no-mlflow
```

### Train on a large volume of viruses

For a strong training signal, fetch hundreds of real sequences (set
`NCBI_EMAIL` first; NCBI requires a contact address). The distance computation is
vectorized and the optimizer samples `optimization.quadruplet_sample_size`
(default 5000) quadruplets per epoch. For very large `n` the per-epoch cost is
dominated by the `n × n` correction update (not the quadruplet count), so lower
`optimization.epochs` (e.g. 200 — best-iterate converges quickly) to keep
training fast.

```bash
export NCBI_EMAIL="you@example.com"
# ~300 Influenza A hemagglutinin sequences (diverse strains, ~1.7 kb each):
python -m src.scraper_ncbi --query "Influenza A virus[Organism] AND hemagglutinin AND 1600:1800[SLEN]" --max-records 300
python main.py --no-mlflow
```

On this 300-sequence set the model converges in ~18 s with a relative
improvement of ~0.999. Raise `--max-records` (and, for very large `n`, keep
`quadruplet_sample_size` bounded) to scale further; the model has also been
trained on 1500 real sequences (relative improvement ~0.999).

### Online learning (improve the model with use)

The correction can keep improving as new data arrives, instead of retraining from
scratch — see [`src/online_learning.py`](src/online_learning.py):

```python
from src.online_learning import OnlineTropicalModel

model = OnlineTropicalModel().fit(names, sequences)   # initial training
model.partial_fit(epochs=200)                          # continue (warm start)
model.add_sequences(new_names, new_sequences)          # incorporate new taxa online
model.save("models/online.npz")                        # persist; resume later with .load()
print(model.score()["relative_improvement"])
```

`add_sequences` embeds the learned `omega` into the enlarged pair space and
warm-starts, so an update is far cheaper than full retraining; the best-iterate
guarantee means continued training never makes the correction worse than before.

Artifacts land in `outputs/` (`distance_matrix.csv`, `corrected_distance_matrix.csv`,
`omega.csv`, `history.csv`, `metrics.json`, `figures/*.png`, `trees/tree_after.{newick,csv,dot}`)
and `models/omega.npy`; a report is written to `reports/evaluation_report.md`.

## Run locally with Docker (recommended)

Docker Compose starts three isolated services on one private network:

| Service | Container role | Host URL |
| --- | --- | --- |
| `frontend` | nginx serves React and reverse-proxies API requests | http://localhost:8080 |
| `backend` | FastAPI only | http://localhost:8000 |
| `streamlit` | Secondary Streamlit dashboard | http://localhost:8501 |

```bash
docker compose build
docker compose up -d
docker compose ps
```

Open **http://localhost:8080** for the professional React application. nginx
serves the SPA and proxies `/health`, `/pipeline`, online-learning, tree, matrix,
and documentation endpoints to `backend:8000` over Docker service-name DNS.
The browser therefore stays on one origin and does not need cross-origin API
calls.

- **http://localhost:8080/docs** — FastAPI Swagger UI through nginx.
- **http://localhost:8000/docs** — direct backend documentation.
- **http://localhost:8501** — Streamlit; its `API_BASE` is
  `http://backend:8000` inside Compose.

Stop with `docker compose down`. `data/`, `outputs/`, and `models/` are mounted
into the Python services so runs and the online model persist on your machine.

> `0.0.0.0` is a server bind address, not a browser destination. Always use
> `http://localhost:<published-port>` from the host.

## Frontend (React + Vite + TypeScript + Tailwind)

The professional UI lives in [`frontend/`](frontend/). In production, the
frontend image builds it with Node 22 and nginx serves `frontend/dist` from `/`.
To build or develop it locally:

```bash
cd frontend
npm install
npm run build      # outputs frontend/dist
npm run dev        # OR a hot-reload dev server (proxies the API at :8000)
```

Development calls `http://localhost:8000` directly. The production build uses
an empty API base, so requests stay on `http://localhost:8080` and nginx proxies
them to FastAPI. The frontend orchestrates the API only — no scientific math is
implemented in React.

The responsive application shell includes grouped scientific navigation,
accessible light and dark themes, explicit loading and error states, mobile
navigation, interactive matrix views, tree controls, and artifact export cards.

## API and dashboards (without Docker)

```bash
uvicorn api.main:app --reload        # React UI (if built) at http://127.0.0.1:8000 ; docs at /docs
streamlit run app/streamlit_app.py   # secondary dashboard at http://127.0.0.1:8501
mlflow ui                            # browse logged runs (after a training run)
```

## Documentation

- [`docs/mathematics.md`](docs/mathematics.md) — the math as implemented (with notes on two deliberate deviations from the draft spec: α-weighted distance; a *normalized* tropical-norm optimizer step, since the literal `γ·‖g‖/√p` diverges).
- [`docs/architecture.md`](docs/architecture.md) — module map and dependency direction.
- [`docs/api.md`](docs/api.md) — endpoint reference.

## Status

Complete: core math, four-point engine, Hybrid tropical gradient descent,
Neighbor-Joining + tree export, orchestration (`predict`/`train`/`evaluate`),
FastAPI (7 endpoints), Streamlit dashboard (10 pages), optional NCBI scraper,
DVC pipeline, MLflow logging, and CI. Full `pytest` suite green.
