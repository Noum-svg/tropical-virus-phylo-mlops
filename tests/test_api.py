"""Tests for :mod:`api.main` -- FastAPI endpoints on small fixtures."""

from __future__ import annotations

from io import StringIO

import numpy as np
from Bio import Phylo
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

TOY_NAMES = ["v1", "v2", "v3", "v4"]
TOY_SEQS = [
    "AUGGUGAAGUACCUAACGUAGCUAGCUA",
    "AUGGUGAAAUAUCUAACGUAGCUAGCUA",
    "ACGUACGUACGUACGUACGUACGUACGU",
    "ACGUACGUACGAACGUACGUUCGUACGU",
]


def line_metric(points) -> list[list[float]]:
    x = np.asarray(points, dtype=float)
    return np.abs(x[:, None] - x[None, :]).tolist()


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)
    assert "timestamp" in body


def test_distance_matrix_endpoint():
    resp = client.post(
        "/distance-matrix", json={"sequences": TOY_SEQS, "virus_names": TOY_NAMES}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["labels"] == TOY_NAMES
    arr = np.array(body["matrix"])
    assert arr.shape == (4, 4)
    np.testing.assert_allclose(np.diag(arr), 0.0)
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_four_point_score_endpoint():
    resp = client.post(
        "/four-point-score", json={"matrix": line_metric([0, 2, 5, 9, 14])}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "n",
        "n_quadruplets",
        "n_sampled",
        "mean_violation",
        "median_violation",
        "max_violation",
        "l2_loss",
        "percent_exact",
    }
    assert body["max_violation"] < 1e-9  # exact tree metric


def test_correct_distance_matrix_endpoint():
    matrix = line_metric([0, 1, 3, 6, 10])
    resp = client.post(
        "/correct-distance-matrix", json={"matrix": matrix, "epochs": 50}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert np.array(body["omega"]).shape == (5, 5)
    assert np.array(body["corrected_matrix"]).shape == (5, 5)
    assert "relative_improvement" in body
    assert body["relative_improvement"] >= -1e-9
    assert isinstance(body["history"], list)


def test_tropical_correction_alias():
    matrix = line_metric([0, 1, 3, 6, 10])
    resp = client.post("/tropical-correction", json={"matrix": matrix, "epochs": 50})
    assert resp.status_code == 200
    assert "corrected_matrix" in resp.json()


def test_phylogenetic_tree_endpoint():
    labels = ["a", "b", "c", "d", "e"]
    resp = client.post(
        "/phylogenetic-tree",
        json={"matrix": line_metric([0, 2, 5, 9, 14]), "labels": labels},
    )
    assert resp.status_code == 200
    body = resp.json()
    parsed = Phylo.read(StringIO(body["newick"]), "newick")
    assert sorted(t.name for t in parsed.get_terminals()) == sorted(labels)
    assert len(body["edges"]) > 0
    assert body["dot"].startswith("digraph")


def test_predict_from_sequences_endpoint():
    resp = client.post(
        "/predict-from-sequences",
        json={
            "sequences": TOY_SEQS,
            "virus_names": TOY_NAMES,
            "epochs": 50,
            "gamma": 0.05,
            "lambda_reg": 0.01,
            "epsilon": 1e-8,
            "quadruplet_sample_size": 500,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["virus_names"] == TOY_NAMES
    assert np.array(body["distance_matrix"]).shape == (4, 4)
    assert np.array(body["corrected_distance_matrix"]).shape == (4, 4)
    parsed = Phylo.read(StringIO(body["tree_newick"]), "newick")
    assert sorted(t.name for t in parsed.get_terminals()) == sorted(TOY_NAMES)
    assert body["relative_improvement"] >= -1e-9


def test_bad_matrix_returns_400():
    resp = client.post("/four-point-score", json={"matrix": [[0, 1, 2], [1, 0, 3]]})
    assert resp.status_code == 400


def test_sequence_name_mismatch_returns_400():
    resp = client.post(
        "/distance-matrix", json={"sequences": TOY_SEQS, "virus_names": ["only_one"]}
    )
    assert resp.status_code == 400


def test_index_serves_spa():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Tropical Virus MLOps" in resp.text


def test_example_csv_download():
    resp = client.get("/example-csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


def test_pipeline_endpoint_on_demo():
    resp = client.post(
        "/pipeline",
        data={"use_demo": "true", "epochs": "40", "min_seq_length": "1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert {"dataset", "distance_matrix", "corrected_matrix", "tree", "history"} <= set(
        body
    )
    assert body["dataset"]["n_valid"] >= 1
    assert body["tree"]["n_leaves"] == body["dataset"]["n_valid"]
    assert body["relative_improvement"] >= -1e-9


def test_pipeline_without_input_returns_400():
    resp = client.post("/pipeline", data={"use_demo": "false"})
    assert resp.status_code == 400


def test_tree_image_returns_png():
    matrix = line_metric([0, 2, 5, 9])
    resp = client.post(
        "/tree-image", json={"matrix": matrix, "labels": ["a", "b", "c", "d"]}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_online_learning_accumulates_and_dedupes(tmp_path, monkeypatch):
    import api.main as apimain

    monkeypatch.setattr(apimain, "ONLINE_MODEL_PATH", tmp_path / "online.npz")

    # first input -> all taxa are new
    r1 = client.post(
        "/online-learn",
        data={"use_demo": "true", "epochs": "40", "min_seq_length": "1"},
    )
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["online"]["total_taxa"] >= 2
    assert b1["online"]["added"] == b1["online"]["total_taxa"]
    assert b1["relative_improvement"] >= -1e-9
    assert "corrected_matrix" in b1 and "tree" in b1

    # same input again -> dedup (0 added), model refined via partial_fit
    b2 = client.post(
        "/online-learn",
        data={"use_demo": "true", "epochs": "40", "min_seq_length": "1"},
    ).json()
    assert b2["online"]["added"] == 0
    assert b2["online"]["total_taxa"] == b1["online"]["total_taxa"]

    # a different input grows the model online
    csv = b"virus_name,rna_sequence\nNEW1,ACGTACGTACGTACGT\nNEW2,ACGTACGAACGTACGT\nNEW3,TTTTACGTACGTTTTT\n"
    b3 = client.post(
        "/online-learn",
        files={"file": ("new.csv", csv, "text/csv")},
        data={"epochs": "40", "min_seq_length": "1"},
    ).json()
    assert b3["online"]["added"] == 3
    assert b3["online"]["total_taxa"] == b1["online"]["total_taxa"] + 3

    # status + reset
    status = client.get("/online-status").json()
    assert status["exists"] and status["total_taxa"] == b3["online"]["total_taxa"]
    assert client.post("/online-reset").json()["reset"] is True
    assert client.get("/online-status").json()["exists"] is False
