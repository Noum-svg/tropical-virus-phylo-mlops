# API reference

FastAPI service in `api/main.py`. Start with:

```bash
uvicorn api.main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

The API only orchestrates `src/`; NumPy/pandas values are converted to JSON
types at the boundary. Errors: 422 (request validation), 400 (semantic, e.g.
non-square matrix or length mismatch).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | `{status, model_loaded, timestamp}` |
| POST | `/distance-matrix` | sequences → `D` (cleaned, labeled) |
| POST | `/four-point-score` | matrix → tropical violation metrics |
| POST | `/correct-distance-matrix` | matrix → `omega`, `X`, metrics_before/after, RI, history |
| POST | `/tropical-correction` | alias of `/correct-distance-matrix` |
| POST | `/phylogenetic-tree` | matrix → tree (Newick / edges / DOT) |
| POST | `/predict-from-sequences` | full pipeline → D, X, omega, metrics, tree |

## `/predict-from-sequences` request

```json
{
  "sequences": ["AUGCUU...", "ACGTTT..."],
  "virus_names": ["virus_1", "virus_2"],
  "alpha": 0.9,
  "epochs": 200,
  "gamma": 0.05,
  "lambda_reg": 0.01,
  "epsilon": 1e-8,
  "quadruplet_sample_size": 500,
  "seed": 42
}
```

Matrix endpoints accept `{"matrix": [[...], ...], "labels": [...]}`; the matrix is
symmetrized and zero-diagonal-enforced defensively before use.

`model_loaded` in `/health` reports whether a trained `omega` artifact
(`models/omega.npy` or `outputs/omega.csv`) was found at startup.
