# Tropical Virus PhyloTree MLOps - Complete Specification

> **Reconciled to the implemented build (v2).** This document has been updated to
> match the code in `src/`. The authoritative, detailed reference lives in
> [`docs/mathematics.md`](docs/mathematics.md), [`docs/architecture.md`](docs/architecture.md),
> and [`docs/api.md`](docs/api.md). Two deliberate departures from the original
> draft are flagged inline below: the distance is `alpha`-weighted (so `d in [0,1]`),
> and the optimizer uses a **normalized** tropical-norm step (the literal
> `gamma*||g||/sqrt(p)` diverges).

## 1. Project Identity

**Project name:** Tropical Virus PhyloTree MLOps

**GitHub repository:**
`https://github.com/Noum-svg/tropical-virus-phylo-mlops`

**Primary objective:** transform a CSV containing viral RNA/DNA sequences into
an optimized phylogenetic tree linking all valid viruses.

The distance matrix is an intermediate artifact. The principal result is the
tree, exported in:

- Newick format;
- edge-list CSV format;
- DOT/Graphviz format;
- PNG and SVG through the Streamlit interface.

## 2. Scientific Scope

The system combines:

1. viral sequence preprocessing;
2. pairwise sequence distance computation;
3. tropical four-point compatibility measurement;
4. NumPy-only Tropical Gradient Descent;
5. Neighbor-Joining tree reconstruction;
6. MLOps delivery through FastAPI, Streamlit, tests, Docker, and GitHub.

This application is for computational biology, research, and education. It is
not a medical diagnostic system and must not be presented as one.

## 3. Input Contract

The required CSV columns are exactly:

```text
virus_name
rna_sequence
```

Example:

```csv
virus_name,rna_sequence
Virus_A,AUGGUGAAGUACCUAACGUAGCUAGCUA
Virus_B,AUGGUGAAAUAUCUAACGUAGCUAGCUA
```

The committed file `data/sample/sample_viral_sequences.csv` is synthetic
demonstration data only. It must never be described as a real biomedical
dataset.

Real production data may come from a curated source such as NCBI, subject to
appropriate provenance, licensing, and privacy controls.

## 4. Sequence Cleaning

For each input sequence:

1. convert to uppercase;
2. replace `U` with `T`;
3. retain only `A`, `C`, `G`, and `T`;
4. reject empty identifiers;
5. reject sequences shorter than `min_length`;
6. remove duplicate `(virus_name, clean_sequence)` rows.

The cleaned DataFrame contains:

```text
virus_name
rna_sequence
clean_sequence
sequence_length
```

## 5. Distance Matrix

Let

$$
\mathcal{S}=\{s_1,\ldots,s_n\}
$$

be the cleaned sequences. The initial distance matrix is

$$
D=(d_{ij})\in\mathbb{R}^{n\times n}.
$$

For two sequences $s_i$ and $s_j$, define

$$
L_{ij}=\min(|s_i|,|s_j|).
$$

The normalized Hamming component is

$$
h(s_i,s_j)
=
\frac{1}{L_{ij}}
\sum_{k=1}^{L_{ij}}
\mathbf{1}[s_i(k)\neq s_j(k)].
$$

The length penalty is

$$
p(s_i,s_j)
=
\frac{\left||s_i|-|s_j|\right|}
{\max(|s_i|,|s_j|)}.
$$

The implemented distance is the convex combination (with `alpha = 0.9` by
default), which keeps `d` in `[0, 1]`:

$$
\boxed{d(s_i,s_j)=\alpha\, h(s_i,s_j)+(1-\alpha)\, p(s_i,s_j)}.
$$

The matrix must satisfy

$$
D=D^\top,\qquad D_{ii}=0,\qquad D_{ij}\geq 0.
$$

## 6. Tropical Four-Point Condition

For every quadruplet

$$
i<j<k<l,
$$

define

$$
S_1=X_{ij}+X_{kl},
$$

$$
S_2=X_{ik}+X_{jl},
$$

$$
S_3=X_{il}+X_{jk}.
$$

The tropical violation is

$$
\boxed{
\delta_{ijkl}(X)
=
\max(S_1,S_2,S_3)
-
\mathrm{secondmax}(S_1,S_2,S_3)
}.
$$

Interpretation:

$$
\delta_{ijkl}(X)=0
\Longrightarrow
\text{four-point compatibility},
$$

$$
\delta_{ijkl}(X)>0
\Longrightarrow
\text{tropical violation}.
$$

Metrics returned by the system:

- `mean_violation`;
- `median_violation`;
- `max_violation`;
- `percent_exact`;
- `loss_l2`;
- `num_quadruplets`.

The squared violation loss is

$$
L_2(X)
=
\sum_{i<j<k<l}
\delta_{ijkl}(X)^2.
$$

For large $n$, quadruplets may be sampled reproducibly with a fixed random
seed.

## 7. Tropical Correction

The learned correction is

$$
\omega\in\mathbb{R}^{n\times n},
$$

and the corrected matrix is

$$
\boxed{X=D+\omega}.
$$

The optimization objective (sum form, as implemented) is

$$
\boxed{
\mathcal{L}(\omega)
=
\sum_{q\in\mathcal{Q}}
\delta_q(D+\omega)^2
+
\lambda\sum_{a,b}\omega_{ab}^2
}.
$$

Required constraints:

$$
\omega=\omega^\top,\qquad \omega_{ii}=0,
$$

$$
X=X^\top,\qquad X_{ii}=0,\qquad X_{ij}\geq 0.
$$

The implementation uses a practical subgradient and the update

$$
\widetilde{\omega}^{(t+1)}
=
\omega^{(t)}
-
\eta_t\nabla_\omega\mathcal{L}(\omega^{(t)}),
$$

Define the tropical gradient spread

$$
T(g)=\max(g)-\min(g).
$$

The step is normalized by this spread (never by the Euclidean norm):

$$
\eta_t=\frac{\gamma}{T(g^{(t)})+\varepsilon}.
$$

After each update the candidate is projected back onto the distance space
(symmetrize, clip negatives so $X\ge 0$, zero the diagonal):

$$
X^{(t+1)}
=
\Pi_{\mathcal{D}_n}
\left(D+\widetilde{\omega}^{(t+1)}\right),
\qquad
\omega^{(t+1)}=X^{(t+1)}-D.
$$

Iteration stops when $T(g)\le\varepsilon$, and the
**best iterate** (lowest loss, including the initial $\omega=0$) is returned so
$X$ is never worse than $D$.

> The literal draft step $\eta_t=\gamma T(g)/(\sqrt{p}+\varepsilon)$
> puts the norm in the numerator and **diverges** (overflows to NaN); the
> normalized form above is used instead. The data-gradient is zero-sum per
> quadruplet, so its tropical norm is a faithful magnitude.

No PyTorch or TensorFlow is permitted.

## 8. Tropical Metric

For vectors $x,y\in\mathbb{R}^m$,

$$
\boxed{
d_{\mathrm{tr}}(x,y)
=
\max_r(x_r-y_r)
-
\min_r(x_r-y_r)
}.
$$

The relative improvement is

$$
\boxed{
\mathrm{relative\_improvement}
=
\frac{L_2(D)-L_2(X)}
{L_2(D)+\varepsilon}
}.
$$

## 9. Neighbor-Joining

The final tree is reconstructed from the corrected matrix $X$.

For an active matrix $M$ with $m$ nodes:

$$
r_i=\sum_k M_{ik},
$$

$$
\boxed{Q_{ij}=(m-2)M_{ij}-r_i-r_j}.
$$

The selected pair is

$$
(i^*,j^*)
=
\arg\min_{i<j}Q_{ij}.
$$

For the new internal node $u$:

$$
\ell_{i^*,u}
=
\frac{1}{2}M_{i^*j^*}
+
\frac{r_{i^*}-r_{j^*}}{2(m-2)},
$$

$$
\ell_{j^*,u}
=
M_{i^*j^*}-\ell_{i^*,u}.
$$

The new cluster distance is

$$
\boxed{
M_{uk}
=
\frac{1}{2}
\left(
M_{i^*k}
+
M_{j^*k}
-
M_{i^*j^*}
\right)
}.
$$

The edge-list schema is:

```json
{
  "parent": "node_1",
  "child": "Virus_A",
  "branch_length": 0.032
}
```

## 10. Pipeline Contract

The complete workflow is:

```text
CSV viral sequences
-> schema validation
-> cleaning and filtering
-> distance matrix D
-> tropical metrics before correction
-> learn omega
-> corrected matrix X = D + omega
-> tropical metrics after correction
-> Neighbor-Joining tree
-> matrices, metrics, history, Newick, edges, DOT
```

The main orchestration function is:

```python
predict_from_sequences(
    virus_names,
    sequences,
    min_length=50,
    epochs=200,
    gamma=0.05,
    lambda_reg=0.01,
    epsilon=1e-8,
    quadruplet_sample_size=500,
)
```

Its return value contains:

```text
virus_names
clean_sequences
sequence_lengths
distance_matrix
corrected_distance_matrix
omega
metrics_before
metrics_after
relative_improvement
tree_newick
tree_edges
tree_dot
history
```

## 11. Repository Structure

```text
.
|-- api/
|   `-- main.py
|-- app/
|   |-- streamlit_app.py
|   `-- ui_helpers.py
|-- data/
|   |-- raw/sample_viral_sequences.csv
|   `-- processed/.gitkeep
|-- docs/
|   |-- api.md
|   |-- architecture.md
|   `-- mathematics.md
|-- outputs/
|   |-- figures/.gitkeep
|   |-- interface_runs/.gitkeep
|   `-- trees/.gitkeep
|-- reports/
|   |-- README.md
|   `-- scientific_report.md
|-- src/
|   |-- data_loader.py
|   |-- distances.py
|   |-- tropical_grassmannian.py
|   |-- tropical_gradient_descent.py
|   |-- phylogeny.py
|   |-- predict.py
|   |-- train.py
|   `-- evaluate.py
|-- tests/
|-- Dockerfile
|-- docker-compose.yml
|-- params.yaml
|-- requirements.txt
|-- README.md
`-- SPEC.md
```

## 12. Streamlit Application

Command:

```bash
streamlit run app/streamlit_app.py
```

Local URL:

```text
http://127.0.0.1:8501
```

Required pages:

1. Overview
2. Dataset
3. Preprocessing
4. Distance Matrix
5. Tropical Correction
6. Phylogenetic Tree
7. Metrics
8. Downloads
9. API Docs
10. About

Required features:

- CSV upload;
- downloadable example CSV;
- data preview and length statistics;
- sequence cleaning;
- pipeline progress indicator;
- distance matrix heatmap, statistics, and distribution;
- training loss chart;
- corrected matrix and omega views;
- circular and rectangular tree layouts;
- label, branch-length, and font-size controls;
- before/after metrics;
- downloads for CSV, JSON, Newick, DOT, PNG, and SVG;
- API health check;
- light and dark themes.

Streamlit must orchestrate functions from `src/`. Mathematical algorithms must
not be duplicated in the UI.

## 13. FastAPI

Command:

```bash
uvicorn api.main:app --reload
```

Documentation:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

```text
GET  /health
POST /distance-matrix
POST /four-point-score
POST /correct-distance-matrix
POST /tropical-correction
POST /predict-from-sequences
POST /phylogenetic-tree
```

The main endpoint is `/predict-from-sequences`.

## 14. Configuration

`params.yaml`:

```yaml
data:
  raw_path: data/sample/sample_viral_sequences.csv
  min_length: 20

optimization:
  epochs: 200
  gamma: 0.05
  lambda_reg: 0.01
  epsilon: 1.0e-8
  quadruplet_sample_size: 500
  random_state: 42

outputs:
  output_dir: outputs
```

## 15. Generated Artifacts

Batch outputs:

```text
outputs/distance_matrix.csv
outputs/corrected_distance_matrix.csv
outputs/omega.csv
outputs/history.csv
outputs/metrics.json
outputs/figures/loss_curve.png
outputs/trees/tree_after.newick
outputs/trees/tree_after_edges.csv
outputs/trees/tree_after.dot
```

Streamlit runs are stored under:

```text
outputs/interface_runs/
```

Generated outputs are ignored by Git.

## 16. Quality And Tests

Run:

```bash
python -m pytest tests
```

Known verified state:

```text
15 passed
```

Tests cover:

- sequence normalization;
- schema validation;
- distance symmetry and zero diagonal;
- non-negative tropical violations;
- tropical metric keys;
- correction constraints;
- Neighbor-Joining outputs;
- complete prediction pipeline;
- FastAPI health and prediction;
- tree image generation and export.

## 17. Docker

Run both services:

```bash
docker compose up --build
```

Services:

```text
FastAPI:   http://127.0.0.1:8000
Streamlit: http://127.0.0.1:8501
```

## 18. GitHub And Publication State

Public repository:

```text
https://github.com/Noum-svg/tropical-virus-phylo-mlops
```

Branch:

```text
main
```

Published commits:

```text
1400b7b Initial commit: Tropical Virus PhyloTree MLOps
b10a331 Add professional scientific documentation
```

Repository description and topics are configured.

The local file `.github/workflows/ci.yml` exists but was not present in the
published tree at the time this specification was written because GitHub
required an additional OAuth `workflow` scope. A future agent should publish
this file after obtaining the scope, without committing secrets.

## 19. Hard Constraints

- Do not use PyTorch.
- Do not use TensorFlow.
- Do not commit private or large datasets.
- Do not commit credentials, tokens, `.env`, logs, or generated outputs.
- Do not claim the sample CSV is real biomedical data.
- Do not duplicate scientific logic in Streamlit or FastAPI.
- Preserve symmetric matrices and zero diagonals.
- Keep distances non-negative.
- Keep code and code comments in English.
- User-facing explanations may be in French.
- Do not change unrelated projects in the parent workspace.

## 20. Definition Of Done

The project is complete when:

- the required CSV is accepted;
- sequences are cleaned and filtered;
- $D$ is constructed;
- tropical metrics before correction are computed;
- $\omega$ is learned with NumPy;
- $X=D+\omega$ is constructed;
- metrics after correction and relative improvement are reported;
- the Neighbor-Joining tree is reconstructed from $X$;
- Newick, edges, DOT, PNG, and SVG are available;
- FastAPI endpoints work;
- all Streamlit pages work;
- downloads work;
- tests pass;
- documentation and equations render clearly on GitHub;
- no secrets or private data are committed.
