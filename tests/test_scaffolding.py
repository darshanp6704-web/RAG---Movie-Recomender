"""Phase 0 scaffolding verification tests."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from src import config


def test_python_version_is_3_11() -> None:
    """Verify the active runtime is Python 3.11.x."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 11


def test_directory_structure_and_files_exist() -> None:
    """Verify required Phase 0 directories, config files, and source workbook exist."""
    base = config.BASE_DIR
    for rel in (
        "data",
        "docs",
        "src",
        "tests",
        "requirements.txt",
        ".gitignore",
        ".env.example",
        "data/movie_buddy_starter_3000.xlsx",
        "src/config.py",
    ):
        assert (base / rel).exists(), f"Missing required path: {rel}"


def test_gitignore_excludes_secrets_and_venv() -> None:
    """Verify .gitignore excludes .env and .venv."""
    gitignore = (config.BASE_DIR / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert ".venv" in gitignore


def test_no_forbidden_heavy_packages_installed() -> None:
    """Verify torch and sentence-transformers are absent to respect 512 MB RAM cap."""
    installed = {
        dist.metadata["Name"].lower()
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    assert "torch" not in installed
    assert "sentence-transformers" not in installed


def test_core_dependencies_importable() -> None:
    """Verify lightweight runtime and ingestion packages import cleanly."""
    import chromadb  # noqa: F401
    import dotenv  # noqa: F401
    import fastapi  # noqa: F401
    import fastembed  # noqa: F401
    import groq  # noqa: F401
    import openpyxl  # noqa: F401
    import pydantic  # noqa: F401
    import uvicorn  # noqa: F401

    assert config.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert Path(config.EXCEL_PATH).exists()
