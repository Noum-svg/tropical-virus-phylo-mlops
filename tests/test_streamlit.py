"""Smoke tests for the Streamlit dashboard via ``AppTest`` (no browser needed)."""

from __future__ import annotations

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = "app/streamlit_app.py"


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception


def test_navigation_pages_render():
    at = AppTest.from_file(APP, default_timeout=30).run()
    for page in ("Dataset", "Preprocessing", "Metrics", "About", "API Docs"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"page {page} raised"
