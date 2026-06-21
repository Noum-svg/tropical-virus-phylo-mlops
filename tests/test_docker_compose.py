"""Regression tests for the multi-container Docker architecture."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_builds_one_image_per_service():
    services = _compose()["services"]

    assert set(services) == {"backend", "frontend", "streamlit"}
    assert services["backend"]["build"]["dockerfile"] == "docker/Dockerfile.backend"
    assert services["frontend"]["build"]["dockerfile"] == "docker/Dockerfile.frontend"
    assert services["streamlit"]["build"]["dockerfile"] == "docker/Dockerfile.streamlit"
    assert len({service["image"] for service in services.values()}) == 3


def test_services_use_healthchecks_and_shared_network():
    compose = _compose()
    services = compose["services"]

    assert "tropical-network" in compose["networks"]
    for service in services.values():
        assert service["restart"] == "unless-stopped"
        assert service["networks"] == ["tropical-network"]
        assert "healthcheck" in service

    assert services["frontend"]["depends_on"]["backend"]["condition"] == (
        "service_healthy"
    )
    assert services["streamlit"]["depends_on"]["backend"]["condition"] == (
        "service_healthy"
    )
    assert services["streamlit"]["environment"]["API_BASE"] == ("http://backend:8000")


def test_nginx_serves_spa_and_proxies_required_api_paths():
    config = (ROOT / "docker" / "nginx.conf").read_text(encoding="utf-8")
    required_paths = {
        "health",
        "pipeline",
        "online-learn",
        "online-status",
        "online-reset",
        "tree-image",
        "example-csv",
        "distance-matrix",
        "four-point-score",
        "correct-distance-matrix",
        "tropical-correction",
        "phylogenetic-tree",
        "predict-from-sequences",
        "docs",
        r"openapi\.json",
    }

    assert "try_files $uri $uri/ /index.html;" in config
    assert "proxy_pass http://backend:8000;" in config
    for path in required_paths:
        assert path in config


def test_vite_builds_assets_for_nginx_root():
    config = (ROOT / "frontend" / "vite.config.ts").read_text(encoding="utf-8")

    assert 'base: "/"' in config
