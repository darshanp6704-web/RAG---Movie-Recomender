"""Phase 1 unit and integration tests for Excel loading, cleaning, and joining."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src import config
from src.ingest import (
    clean_certificate,
    clean_title_row,
    clean_watch_row,
    load_and_clean_catalog,
)


def test_empty_and_whitespace_certificate_becomes_unknown_or_trimmed() -> None:
    """Verify EC-ING-01 and EC-ING-02 certificate normalization."""
    assert clean_certificate(None) == "Unknown"
    assert clean_certificate("") == "Unknown"
    assert clean_certificate("   ") == "Unknown"
    assert clean_certificate("UA ") == "UA"


def test_clean_title_row_preserves_types_and_trims_text() -> None:
    """Verify EC-ING-03, EC-ING-04, EC-ING-05 type coercion and trimming."""
    row = clean_title_row(
        {
            "title_id": " m_1 ",
            "type": "Movie ",
            "title": " Sample ",
            "release_year": "2020",
            "runtime_min": " 120 ",
            "seasons": None,
            "rating": 8.199999999999999,
            "vote_count": "150",
            "popularity": "42.5",
            "free_to_watch": "1",
            "has_plot": "0",
            "certification_in": " ",
            "genres": " Drama , Comedy , ",
        }
    )
    assert row["title_id"] == "m_1"
    assert row["type"] == "Movie"
    assert row["title"] == "Sample"
    assert row["release_year"] == 2020 and isinstance(row["release_year"], int)
    assert row["runtime_min"] == 120 and isinstance(row["runtime_min"], int)
    assert row["seasons"] is None
    assert row["rating"] == 8.2 and isinstance(row["rating"], float)
    assert row["vote_count"] == 150 and isinstance(row["vote_count"], int)
    assert row["popularity"] == 42.5 and isinstance(row["popularity"], float)
    assert row["free_to_watch"] is True
    assert row["has_plot"] is False
    assert row["certification_in"] == "Unknown"
    assert row["genres"] == "Drama, Comedy"


def test_clean_watch_row_uses_platform_group() -> None:
    """Verify platform_group is used instead of granular platform name."""
    entry = clean_watch_row(
        {
            "title_id": "m_148265",
            "title": "Eega",
            "platform": "ManoramaMAX Amazon Channel",
            "platform_group": "Prime Video add on",
            "access_type": "Subscription",
            "region": "IN",
            "as_of": "2026-11-21",
        }
    )
    assert entry["platform"] == "Prime Video add on"
    assert entry["access_type"] == "Subscription"


def test_processed_titles_json_and_sqlite_outputs() -> None:
    """Verify generated data/processed/titles.json and SQLite DB integrity."""
    json_path = config.PROCESSED_TITLES_PATH
    db_path = config.SQLITE_DB_PATH
    assert json_path.exists(), "Run `python -m src.ingest` before testing outputs."
    assert db_path.exists(), "SQLite DB should exist after Phase 1 ingestion."

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert len(data) == 3000

    # Verify no empty or whitespace-padded certification_in
    certs = [row["certification_in"] for row in data]
    assert all(isinstance(c, str) and c != "" and c == c.strip() for c in certs)
    assert certs.count("Unknown") == 1226

    # Verify platform_group used for Eega (m_148265) and deduplicated
    eega = next(row for row in data if row["title_id"] == "m_148265")
    platforms = [opt["platform"] for opt in eega["where_to_watch"]]
    assert "Prime Video add on" in platforms
    assert "ManoramaMAX Amazon Channel" not in platforms
    assert len(platforms) == len(
        {
            (opt["platform"], opt["access_type"], opt["region"], opt["as_of"])
            for opt in eega["where_to_watch"]
        }
    )

    # Verify SQLite row counts match
    conn = sqlite3.connect(db_path)
    try:
        title_count = conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
        null_runtime_zero = conn.execute(
            "SELECT COUNT(*) FROM titles WHERE runtime_min = 0"
        ).fetchone()[0]
        assert title_count == 3000
        assert null_runtime_zero == 0
    finally:
        conn.close()


def test_workbook_loads_cleanly_from_source() -> None:
    """Verify load_and_clean_catalog runs cleanly against starter workbook."""
    source_path = config.BASE_DIR / "movie_buddy_starter_3000 (1).xlsx"
    titles, raw_wtw = load_and_clean_catalog(source_path)
    assert len(titles) == 3000
    assert len(raw_wtw) == 8886
