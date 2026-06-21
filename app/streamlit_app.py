"""Streamlit dashboard for the Tropical Virus PhyloTree pipeline.

Orchestrates ``src/`` only -- no distance, four-point, optimization, or
Neighbor-Joining math is implemented here. Run with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app import ui_helpers as ui
from src.data_loader import REQUIRED_COLUMNS, clean_dataframe
from src.distances import distance_matrix_from_clean_df
from src.phylogeny import reconstruct_tree
from src.tropical_gradient_descent import correct_distance_matrix
from src.tropical_grassmannian import tropical_score
from src.utils import matrix_to_vector

DEMO_CSV = Path("data/sample/sample_viral_sequences.csv")
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
PAGES = [
    "Overview",
    "Dataset",
    "Preprocessing",
    "Distance Matrix",
    "Tropical Correction",
    "Phylogenetic Tree",
    "Metrics",
    "Downloads",
    "API Docs",
    "About",
]

DISCLAIMER = (
    "Non-diagnostic research tool. It analyzes viral sequence relationships and "
    "reconstructs trees; it does not diagnose infection or disease."
)

_CSS = """
<style>
section[data-testid="stSidebar"] { background-color: #11161c; }
section[data-testid="stSidebar"] * { color: #e6edf3; }
div[data-testid="stMetric"] {
  background: rgba(127,127,127,0.08); border-radius: 10px; padding: 10px;
}
</style>
"""


def _init_state() -> None:
    for key in ("raw_df", "clean_df", "distance_matrix", "correction", "tree"):
        st.session_state.setdefault(key, None)


def _require(key: str, msg: str) -> bool:
    if st.session_state.get(key) is None:
        st.info(msg)
        return False
    return True


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.title("🧬 Tropical Virus PhyloTree MLOps")
    st.caption(DISCLAIMER)
    st.markdown(
        "Learn a tropical correction **X = D + ω** of a sequence distance matrix, "
        "then reconstruct a Neighbor-Joining tree. The tree is the deliverable; "
        "the distance matrix is an internal artifact."
    )
    corr = st.session_state.get("correction")
    c1, c2, c3 = st.columns(3)
    n = (
        len(st.session_state["clean_df"])
        if st.session_state.get("clean_df") is not None
        else 0
    )
    c1.metric("Sequences", n)
    if corr is not None:
        c2.metric("Relative improvement", f"{corr['relative_improvement']:.3f}")
        c3.metric("L2 after", f"{corr['metrics_after']['l2_loss']:.4g}")
    else:
        c2.metric("Relative improvement", "—")
        c3.metric("L2 after", "—")
    st.markdown(
        "Use the sidebar to walk the pipeline: Dataset → Preprocessing → "
        "Distance Matrix → Tropical Correction → Phylogenetic Tree."
    )


def page_dataset() -> None:
    st.header("Dataset")
    st.write(f"Required columns: `{', '.join(REQUIRED_COLUMNS)}`")
    if DEMO_CSV.exists():
        st.download_button(
            "⬇️ Download example CSV (synthetic demo)",
            DEMO_CSV.read_bytes(),
            file_name="sample_viral_sequences.csv",
            mime="text/csv",
        )
        st.caption("The example is SYNTHETIC demonstration data, not real viral data.")

    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    use_demo = st.checkbox("Use the bundled synthetic demo instead", value=False)

    df = None
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif use_demo and DEMO_CSV.exists():
        df = pd.read_csv(DEMO_CSV)

    if df is not None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing required column(s): {missing}")
        else:
            st.session_state["raw_df"] = df
            st.success(f"Loaded {len(df)} rows.")
            st.dataframe(df.head(20), use_container_width=True)


def page_preprocessing() -> None:
    st.header("Preprocessing")
    if not _require("raw_df", "Load a dataset first (Dataset page)."):
        return
    min_len = st.number_input("Minimum cleaned length", 1, 100000, 1)
    if st.button("Clean & filter"):
        clean_df = clean_dataframe(
            st.session_state["raw_df"], min_seq_length=int(min_len)
        )
        st.session_state["clean_df"] = clean_df
        # invalidate downstream artifacts
        st.session_state["distance_matrix"] = None
        st.session_state["correction"] = None
        st.session_state["tree"] = None
    clean_df = st.session_state.get("clean_df")
    if clean_df is not None:
        st.success(f"{len(clean_df)} sequences after cleaning.")
        st.dataframe(clean_df.head(20), use_container_width=True)
        st.subheader("Sequence length statistics")
        st.write(clean_df["sequence_length"].describe())
        st.pyplot(
            ui.distribution(clean_df["sequence_length"], "Sequence lengths", "length")
        )


def page_distance_matrix() -> None:
    st.header("Distance Matrix D")
    if not _require("clean_df", "Clean a dataset first (Preprocessing page)."):
        return
    alpha = st.slider("alpha (Hamming vs length weight)", 0.0, 1.0, 0.9, 0.05)
    if st.button("Build D"):
        frame = distance_matrix_from_clean_df(st.session_state["clean_df"], alpha=alpha)
        st.session_state["distance_matrix"] = frame
        st.session_state["correction"] = None
        st.session_state["tree"] = None
    frame = st.session_state.get("distance_matrix")
    if frame is not None:
        st.pyplot(ui.heatmap(frame, "Distance matrix D"))
        st.subheader("Distribution of pairwise distances")
        st.pyplot(ui.distribution(matrix_to_vector(frame.to_numpy()), "Distances", "d"))
        score = tropical_score(frame.to_numpy())
        st.subheader("Four-point metrics (before correction)")
        st.json(score)


def page_correction() -> None:
    st.header("Tropical Correction (X = D + ω)")
    if not _require("distance_matrix", "Build D first (Distance Matrix page)."):
        return
    col = st.columns(4)
    epochs = col[0].number_input("epochs", 1, 100000, 200)
    gamma = col[1].number_input("gamma", 0.0, 10.0, 0.05, format="%.4f")
    lambda_reg = col[2].number_input("lambda_reg", 0.0, 10.0, 0.001, format="%.4f")
    sample = col[3].number_input("quad sample (0 = all)", 0, 10_000_000, 0)
    if st.button("Run Tropical Gradient Descent"):
        config = {
            "epochs": int(epochs),
            "gamma": float(gamma),
            "lambda_reg": float(lambda_reg),
            "epsilon": 1e-9,
            "quadruplet_sample_size": int(sample) or None,
            "seed": 42,
        }
        with st.spinner("Optimizing…"):
            result = correct_distance_matrix(
                st.session_state["distance_matrix"].to_numpy(), config
            )
        st.session_state["correction"] = result
        st.session_state["tree"] = None
    corr = st.session_state.get("correction")
    if corr is not None:
        st.success(f"Relative improvement: {corr['relative_improvement']:.4f}")
        st.pyplot(ui.plot_loss_curve(corr["history"]))
        names = list(st.session_state["distance_matrix"].index)
        st.subheader("Corrected matrix X")
        st.pyplot(ui.heatmap(pd.DataFrame(corr["X"], index=names, columns=names), "X"))
        st.subheader("Correction ω")
        st.pyplot(
            ui.heatmap(pd.DataFrame(corr["omega"], index=names, columns=names), "ω")
        )


def page_tree() -> None:
    st.header("Phylogenetic Tree")
    if not _require(
        "correction", "Run the correction first (Tropical Correction page)."
    ):
        return
    names = list(st.session_state["distance_matrix"].index)
    corrected = pd.DataFrame(
        st.session_state["correction"]["X"], index=names, columns=names
    )
    tree = reconstruct_tree(corrected)
    st.session_state["tree"] = tree
    layout = st.radio("Layout", ["rectangular", "circular"], horizontal=True)
    if layout == "rectangular":
        st.pyplot(ui.draw_tree_rectangular(tree.to_newick()))
    else:
        st.pyplot(ui.draw_tree_circular(tree))
    st.subheader("Newick")
    st.code(tree.to_newick(), language="text")


def page_metrics() -> None:
    st.header("Metrics: before vs after")
    if not _require(
        "correction", "Run the correction first (Tropical Correction page)."
    ):
        return
    corr = st.session_state["correction"]
    before, after = corr["metrics_before"], corr["metrics_after"]
    table = pd.DataFrame({"before (D)": before, "after (X)": after})
    st.dataframe(table, use_container_width=True)
    st.metric("Relative improvement", f"{corr['relative_improvement']:.4f}")
    st.pyplot(ui.plot_before_after_summary(before, after))


def _csv_bytes(frame: pd.DataFrame, index: bool = True) -> bytes:
    return frame.to_csv(index=index).encode("utf-8")


def page_downloads() -> None:
    st.header("Downloads")
    if not _require("correction", "Run the pipeline first to enable downloads."):
        return
    names = list(st.session_state["distance_matrix"].index)
    corr = st.session_state["correction"]
    tree = st.session_state.get("tree") or reconstruct_tree(
        pd.DataFrame(corr["X"], index=names, columns=names)
    )
    d = st.session_state["distance_matrix"]
    x = pd.DataFrame(corr["X"], index=names, columns=names)
    omega = pd.DataFrame(corr["omega"], index=names, columns=names)
    metrics = {
        "metrics_before": corr["metrics_before"],
        "metrics_after": corr["metrics_after"],
        "relative_improvement": corr["relative_improvement"],
    }
    st.download_button("distance_matrix.csv", _csv_bytes(d), "distance_matrix.csv")
    st.download_button(
        "corrected_distance_matrix.csv", _csv_bytes(x), "corrected_distance_matrix.csv"
    )
    st.download_button("omega.csv", _csv_bytes(omega), "omega.csv")
    st.download_button(
        "metrics.json", json.dumps(metrics, indent=2).encode(), "metrics.json"
    )
    st.download_button(
        "history.csv", _csv_bytes(corr["history"], index=False), "history.csv"
    )
    st.download_button("tree.newick", tree.to_newick().encode(), "tree.newick")
    st.download_button(
        "tree_edges.csv",
        _csv_bytes(tree.to_edge_dataframe(), index=False),
        "tree_edges.csv",
    )
    st.download_button("tree.dot", tree.to_dot().encode(), "tree.dot")


def page_api_docs() -> None:
    st.header("API")
    st.markdown(
        "The API base URL is configured with `API_BASE`. In Docker, Streamlit "
        "uses the internal service address `http://backend:8000`."
    )
    url = st.text_input("Health endpoint", f"{API_BASE}/health")
    if st.button("Check API health"):
        try:
            import requests

            resp = requests.get(url, timeout=3)
            st.json(resp.json())
        except Exception as exc:  # noqa: BLE001
            st.error(f"API not reachable: {exc}")


def page_about() -> None:
    st.header("About")
    st.markdown(
        "**Tropical Virus PhyloTree MLOps** learns a tropical correction "
        "**X = D + ω** that reduces tropical four-point violations, then runs "
        "Neighbor-Joining.\n\n"
        f"> {DISCLAIMER}\n\n"
        "Reducing four-point violations improves mathematical compatibility with "
        "an additive tree metric; it does not establish biological correctness."
    )


PAGE_FUNCS = {
    "Overview": page_overview,
    "Dataset": page_dataset,
    "Preprocessing": page_preprocessing,
    "Distance Matrix": page_distance_matrix,
    "Tropical Correction": page_correction,
    "Phylogenetic Tree": page_tree,
    "Metrics": page_metrics,
    "Downloads": page_downloads,
    "API Docs": page_api_docs,
    "About": page_about,
}


def main() -> None:
    st.set_page_config(
        page_title="Tropical Virus PhyloTree", page_icon="🧬", layout="wide"
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()
    st.sidebar.title("🧬 PhyloTree")
    st.sidebar.caption("Tropical four-point correction + Neighbor-Joining")
    choice = st.sidebar.radio("Navigate", PAGES)
    PAGE_FUNCS[choice]()


# Streamlit runs the script with __name__ == "__main__" (both `streamlit run`
# and AppTest), so this still renders there, while a plain `import` does not.
if __name__ == "__main__":
    main()
