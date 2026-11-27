"""Phase 9 Render deployment readiness and lazy-loading tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src import config
from src.retriever import _load_embedder_singleton, is_embedder_loaded
from src.ui.server import app


def test_requirements_exclude_heavy_and_nvidia_packages() -> None:
    """Verify requirements.txt excludes torch, sentence-transformers, and nvidia packages."""
    reqs = (config.BASE_DIR / "requirements.txt").read_text(encoding="utf-8").lower()
    for line in reqs.splitlines():
        pkg = line.strip().split(">=")[0].split("==")[0].split("[")[0]
        assert pkg != "torch"
        assert pkg != "sentence-transformers"
        assert not pkg.startswith("nvidia")


def test_render_yaml_and_python_version_configured() -> None:
    """Verify render.yaml binds 0.0.0.0 on $PORT with 1 worker and uses committed chroma_db."""
    render_yaml = (config.BASE_DIR / "render.yaml").read_text(encoding="utf-8")
    assert "plan: free" in render_yaml
    assert "--host 0.0.0.0" in render_yaml
    assert "--port $PORT" in render_yaml
    assert "--workers 1" in render_yaml
    assert "healthCheckPath: /health" in render_yaml
    assert "src.ingest" not in render_yaml
    assert "src.index" not in render_yaml

    py_ver = (config.BASE_DIR / ".python-version").read_text(encoding="utf-8").strip()
    assert py_ver.startswith("3.11")
    assert config.CHROMA_DIR.exists()
    assert (config.CHROMA_DIR / "chroma.sqlite3").exists()


def test_embedder_loads_lazily_not_at_boot_or_health() -> None:
    """Verify FastEmbed is not loaded on app boot, /, or /health."""
    _load_embedder_singleton.cache_clear()
    assert is_embedder_loaded() is False

    with TestClient(app) as client:
        r_idx = client.get("/")
        assert r_idx.status_code == 200
        assert is_embedder_loaded() is False

        r_health = client.get("/health")
        assert r_health.status_code == 200
        data = r_health.json()
        assert data["status"] == "ok"
        assert data["titles_indexed"] == 3000
        assert data["embedder_loaded"] is False
        assert is_embedder_loaded() is False
