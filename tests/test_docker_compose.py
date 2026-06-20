"""Regression tests for the shared Docker Compose application image."""

from __future__ import annotations

from pathlib import Path

import yaml


def _services() -> dict:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    return compose["services"]


def test_compose_builds_the_shared_image_once():
    services = _services()

    assert services["api"]["build"] == "."
    assert "build" not in services["streamlit"]
    assert services["api"]["image"] == services["streamlit"]["image"]


def test_streamlit_uses_python_module_launch():
    command = _services()["streamlit"]["command"]

    assert command.strip().startswith("python -m streamlit run")
