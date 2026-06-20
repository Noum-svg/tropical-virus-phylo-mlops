"""FastAPI service for the Tropical Virus PhyloTree pipeline.

Non-diagnostic research tool: it analyzes viral sequence relationships and
reconstructs trees. It does not diagnose infection or disease.

Every endpoint only orchestrates the scientific core in :mod:`src`; no distance,
four-point, optimization, or Neighbor-Joining math is reimplemented here. NumPy
arrays and pandas frames are converted to plain JSON types at the boundary.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data_loader import REQUIRED_COLUMNS, clean_dataframe, sequence_summary
from src.distances import distance_matrix_from_clean_df
from src.phylogeny import reconstruct_tree
from src.predict import predict_from_sequences
from src.tropical_grassmannian import tropical_score
from src.tropical_gradient_descent import correct_distance_matrix
from src.utils import num_pairs, project_to_distance_space
from src.visualization import render_tree

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
# Prefer the built React app; fall back to the no-build static UI in web/.
UI_DIR = FRONTEND_DIST if (FRONTEND_DIST / "index.html").exists() else WEB_DIR

app = FastAPI(
    title="Tropical Virus PhyloTree MLOps",
    version="1.0.0",
    description=(
        "Learn a tropical correction X = D + omega of a viral sequence distance "
        "matrix and reconstruct a Neighbor-Joining tree. Non-diagnostic research "
        "tool; not a medical device."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the single-page dashboard."""
    idx = UI_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>Tropical Virus PhyloTree API</h1>"
        "<p>UI not built. See <a href='/docs'>/docs</a>.</p>"
    )


# --------------------------------------------------------------------------- #
# Model loading (for /health.model_loaded)
# --------------------------------------------------------------------------- #
def _load_stored_omega() -> np.ndarray | None:
    """Load a previously trained omega from models/ or outputs/, if present."""
    for path in (Path("models/omega.npy"), Path("outputs/omega.csv")):
        if path.exists():
            try:
                if path.suffix == ".npy":
                    return np.load(path)
                return pd.read_csv(path, index_col=0).to_numpy(dtype=float)
            except Exception:  # pragma: no cover - corrupt artifact is non-fatal
                return None
    return None


STORED_OMEGA: np.ndarray | None = _load_stored_omega()


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class SequencesRequest(BaseModel):
    sequences: list[str]
    virus_names: list[str] | None = None
    alpha: float = 0.9
    min_seq_length: int = 1


class MatrixRequest(BaseModel):
    matrix: list[list[float]]
    labels: list[str] | None = None


class FourPointRequest(MatrixRequest):
    eps: float = 1e-9
    sample_size: int | None = None
    seed: int = 42


class CorrectionRequest(MatrixRequest):
    epochs: int = 200
    gamma: float = 0.05
    lambda_reg: float = 0.01
    epsilon: float = 1e-8
    quadruplet_sample_size: int | None = 500
    seed: int = 42


class PredictRequest(BaseModel):
    sequences: list[str]
    virus_names: list[str] | None = None
    alpha: float = 0.9
    min_seq_length: int = 1
    epochs: int = 200
    gamma: float = 0.05
    lambda_reg: float = 0.01
    epsilon: float = 1e-8
    quadruplet_sample_size: int | None = 500
    seed: int = 42


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class DistanceMatrixResponse(BaseModel):
    labels: list[str]
    matrix: list[list[float]]


class FourPointResponse(BaseModel):
    n: int
    n_quadruplets: int
    n_sampled: int
    mean_violation: float
    median_violation: float
    max_violation: float
    l2_loss: float
    percent_exact: float


class CorrectionResponse(BaseModel):
    labels: list[str]
    omega: list[list[float]]
    corrected_matrix: list[list[float]]
    metrics_before: FourPointResponse
    metrics_after: FourPointResponse
    relative_improvement: float
    history: list[dict[str, float]]


class TreeResponse(BaseModel):
    newick: str
    edges: list[dict[str, Any]]
    dot: str


class PredictResponse(BaseModel):
    virus_names: list[str]
    clean_sequences: list[str]
    sequence_lengths: list[int]
    distance_matrix: list[list[float]]
    corrected_distance_matrix: list[list[float]]
    omega: list[list[float]]
    metrics_before: FourPointResponse
    metrics_after: FourPointResponse
    relative_improvement: float
    tree_newick: str
    tree_edges: list[dict[str, Any]]
    tree_dot: str
    history: list[dict[str, float]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-native records (no NumPy scalar types)."""
    return json.loads(df.to_json(orient="records"))


def _matrix_and_labels(req: MatrixRequest) -> tuple[np.ndarray, list[str]]:
    """Validate a square matrix request and enforce symmetry + zero diagonal."""
    arr = np.array(req.matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise HTTPException(400, "matrix must be a non-empty square n x n array.")
    n = arr.shape[0]
    labels = req.labels or [f"taxon_{i}" for i in range(n)]
    if len(labels) != n:
        raise HTTPException(400, f"labels length {len(labels)} != matrix size {n}.")
    # Defensive: symmetrize and zero the diagonal (no-op for valid inputs).
    arr = project_to_distance_space(arr, non_negative=False)
    return arr, labels


def _sequences_to_clean_df(
    sequences: list[str], virus_names: list[str] | None, min_seq_length: int
) -> pd.DataFrame:
    names = virus_names or [f"virus_{i + 1}" for i in range(len(sequences))]
    if len(names) != len(sequences):
        raise HTTPException(
            400, f"virus_names ({len(names)}) != sequences ({len(sequences)})."
        )
    raw = pd.DataFrame({"virus_name": names, "rna_sequence": sequences})
    clean_df = clean_dataframe(raw, min_seq_length=min_seq_length)
    if len(clean_df) == 0:
        raise HTTPException(400, "No sequence survived cleaning/filtering.")
    return clean_df


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check and whether a trained omega artifact is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=STORED_OMEGA is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/distance-matrix", response_model=DistanceMatrixResponse)
def distance_matrix(req: SequencesRequest) -> DistanceMatrixResponse:
    """Clean sequences and return the symmetric distance matrix ``D``."""
    try:
        clean_df = _sequences_to_clean_df(
            req.sequences, req.virus_names, req.min_seq_length
        )
        frame = distance_matrix_from_clean_df(clean_df, alpha=req.alpha)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return DistanceMatrixResponse(
        labels=list(frame.index), matrix=frame.to_numpy().tolist()
    )


@app.post("/four-point-score", response_model=FourPointResponse)
def four_point(req: FourPointRequest) -> FourPointResponse:
    """Return tropical four-point violation metrics for a matrix."""
    arr, _ = _matrix_and_labels(req)
    try:
        score = tropical_score(
            arr, eps=req.eps, sample_size=req.sample_size, seed=req.seed
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return FourPointResponse(**score)


def _run_correction(req: CorrectionRequest) -> CorrectionResponse:
    arr, labels = _matrix_and_labels(req)
    config = {
        "epochs": req.epochs,
        "gamma": req.gamma,
        "lambda_reg": req.lambda_reg,
        "epsilon": req.epsilon,
        "quadruplet_sample_size": req.quadruplet_sample_size,
        "seed": req.seed,
    }
    try:
        result = correct_distance_matrix(arr, config)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return CorrectionResponse(
        labels=labels,
        omega=np.asarray(result["omega"]).tolist(),
        corrected_matrix=np.asarray(result["X"]).tolist(),
        metrics_before=result["metrics_before"],
        metrics_after=result["metrics_after"],
        relative_improvement=result["relative_improvement"],
        history=_records(result["history"]),
    )


@app.post("/correct-distance-matrix", response_model=CorrectionResponse)
def correct(req: CorrectionRequest) -> CorrectionResponse:
    """Learn omega for a matrix and report before/after metrics and improvement."""
    return _run_correction(req)


@app.post("/tropical-correction", response_model=CorrectionResponse)
def tropical_correction(req: CorrectionRequest) -> CorrectionResponse:
    """Alias of ``/correct-distance-matrix`` (kept for API compatibility)."""
    return _run_correction(req)


@app.post("/phylogenetic-tree", response_model=TreeResponse)
def phylogenetic_tree(req: MatrixRequest) -> TreeResponse:
    """Reconstruct a Neighbor-Joining tree from a matrix."""
    arr, labels = _matrix_and_labels(req)
    frame = pd.DataFrame(arr, index=labels, columns=labels)
    tree = reconstruct_tree(frame)
    return TreeResponse(
        newick=tree.to_newick(), edges=tree.to_edge_list(), dot=tree.to_dot()
    )


@app.post("/predict-from-sequences", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Run the full pipeline: sequences -> D -> X -> tree."""
    names = req.virus_names or [f"virus_{i + 1}" for i in range(len(req.sequences))]
    try:
        result = predict_from_sequences(
            names,
            req.sequences,
            min_seq_length=req.min_seq_length,
            alpha=req.alpha,
            epochs=req.epochs,
            gamma=req.gamma,
            lambda_reg=req.lambda_reg,
            epsilon=req.epsilon,
            quadruplet_sample_size=req.quadruplet_sample_size,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return PredictResponse(
        virus_names=result["virus_names"],
        clean_sequences=result["clean_sequences"],
        sequence_lengths=result["sequence_lengths"],
        distance_matrix=result["distance_matrix"].to_numpy().tolist(),
        corrected_distance_matrix=result["corrected_distance_matrix"]
        .to_numpy()
        .tolist(),
        omega=result["omega"].to_numpy().tolist(),
        metrics_before=result["metrics_before"],
        metrics_after=result["metrics_after"],
        relative_improvement=result["relative_improvement"],
        tree_newick=result["tree_newick"],
        tree_edges=result["tree_edges"],
        tree_dot=result["tree_dot"],
        history=_records(result["history"]),
    )


# --------------------------------------------------------------------------- #
# Dashboard endpoints (CSV upload -> full pipeline; tree image)
# --------------------------------------------------------------------------- #
DEMO_CSV = Path("data/sample/sample_viral_sequences.csv")


@app.get("/example-csv")
def example_csv() -> FileResponse:
    """Download the bundled synthetic demonstration CSV."""
    if not DEMO_CSV.exists():
        raise HTTPException(404, "Demo CSV not found.")
    return FileResponse(
        str(DEMO_CSV), media_type="text/csv", filename="sample_viral_sequences.csv"
    )


def _load_input_csv(
    file: UploadFile | None, use_demo: bool
) -> tuple[pd.DataFrame, str]:
    if file is not None and file.filename:
        raw = pd.read_csv(io.BytesIO(file.file.read()))
        return raw, file.filename
    if use_demo:
        if not DEMO_CSV.exists():
            raise HTTPException(404, "Demo CSV not found.")
        return pd.read_csv(DEMO_CSV), DEMO_CSV.name
    raise HTTPException(400, "Provide a CSV file or set use_demo=true.")


@app.post("/pipeline")
def pipeline(
    file: UploadFile | None = File(default=None),
    use_demo: bool = Form(default=False),
    alpha: float = Form(default=0.9),
    min_seq_length: int = Form(default=1),
    epochs: int = Form(default=200),
    gamma: float = Form(default=0.05),
    lambda_reg: float = Form(default=0.01),
    epsilon: float = Form(default=1e-8),
    quadruplet_sample_size: int = Form(default=500),
) -> dict[str, Any]:
    """Run the full pipeline from an uploaded CSV (or the demo) for the dashboard."""
    raw, file_name = _load_input_csv(file, use_demo)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise HTTPException(400, f"CSV missing required column(s): {missing}.")

    names = list(raw["virus_name"].astype(str))
    seqs = list(raw["rna_sequence"].astype(str))
    try:
        result = predict_from_sequences(
            names,
            seqs,
            min_seq_length=min_seq_length,
            alpha=alpha,
            epochs=epochs,
            gamma=gamma,
            lambda_reg=lambda_reg,
            epsilon=epsilon,
            quadruplet_sample_size=(quadruplet_sample_size or None),
        )
        clean_df = clean_dataframe(raw, min_seq_length=min_seq_length)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    summ = sequence_summary(clean_df)
    tree_summary = result["tree"].summary()
    labels = result["virus_names"]
    is_rna = any("U" in s.upper() for s in seqs)
    preview = [
        {"virus_name": n, "rna_sequence": (s[:40] + "…") if len(s) > 40 else s}
        for n, s in zip(names[:5], seqs[:5])
    ]

    return {
        "dataset": {
            "file_name": file_name,
            "type": "RNA" if is_rna else "DNA",
            "n_total": int(len(raw)),
            "n_valid": summ["n_sequences"],
            "min_length": summ["min_length"],
            "max_length": summ["max_length"],
            "mean_length": round(float(summ["mean_length"]), 1),
            "gc_content": round(float(summ["gc_content"]) * 100, 1),
            "preview": preview,
            "total_rows": int(len(raw)),
        },
        "pairs": num_pairs(len(labels)),
        "labels": labels,
        "sequence_lengths": result["sequence_lengths"],
        "distance_matrix": result["distance_matrix"].to_numpy().tolist(),
        "corrected_matrix": result["corrected_distance_matrix"].to_numpy().tolist(),
        "omega": result["omega"].to_numpy().tolist(),
        "metrics_before": result["metrics_before"],
        "metrics_after": result["metrics_after"],
        "relative_improvement": result["relative_improvement"],
        "history": _records(result["history"]),
        "tree": {
            "newick": result["tree_newick"],
            "edges": result["tree_edges"],
            "dot": result["tree_dot"],
            "n_leaves": tree_summary["n_leaves"],
            "n_internal": tree_summary["n_internal"],
            "total_branch_length": round(tree_summary["total_branch_length"], 3),
        },
        "params": {
            "epochs": epochs,
            "gamma": gamma,
            "lambda_reg": lambda_reg,
            "epsilon": epsilon,
            "quadruplet_sample_size": quadruplet_sample_size,
            "optimizer": "Tropical Gradient Descent",
        },
    }


class TreeImageRequest(MatrixRequest):
    layout: str = "circular"
    show_labels: bool = True
    label_size: int = 8
    use_branch_length: bool = True


@app.post("/tree-image")
def tree_image(req: TreeImageRequest) -> StreamingResponse:
    """Render a clade-coloured tree image (PNG) for a matrix."""
    import matplotlib.pyplot as plt

    arr, labels = _matrix_and_labels(req)
    frame = pd.DataFrame(arr, index=labels, columns=labels)
    tree = reconstruct_tree(frame)
    fig = render_tree(
        tree,
        layout=req.layout if req.layout in ("circular", "rectangular") else "circular",
        show_labels=req.show_labels,
        label_size=max(4, min(int(req.label_size), 20)),
        use_branch_length=req.use_branch_length,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
