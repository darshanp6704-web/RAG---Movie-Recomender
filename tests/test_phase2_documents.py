"""Phase 2 tests for atomic document construction, metadata scalars, and outputs."""

from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.documents import build_documents


def test_build_documents_skips_missing_plot_and_preserves_text() -> None:
    """Verify has_plot gate and unmodified search_text."""
    sample_titles = [
        {
            "title_id": "m_1",
            "type": "Movie",
            "language": "Hindi",
            "release_year": 2020,
            "seasons": None,
            "free_to_watch": True,
            "has_plot": True,
            "search_text": "Sample (2020). Hindi Film. Great story.",
        },
        {
            "title_id": "m_2",
            "type": "Movie",
            "has_plot": False,
            "search_text": "Should be skipped.",
        },
    ]
    docs, audit = build_documents(sample_titles)
    assert len(docs) == 1
    assert audit["skipped_count"] == 1
    assert docs[0]["id"] == "m_1"
    assert docs[0]["text"] == "Sample (2020). Hindi Film. Great story."
    assert "seasons" not in docs[0]["metadata"]
    assert docs[0]["metadata"]["free_to_watch"] is True


def test_generated_documents_files_match_titles_exactly() -> None:
    """Verify documents.jsonl and documents.json match titles.json without altering data."""
    titles_path = config.PROCESSED_TITLES_PATH
    jsonl_path = config.PROCESSED_DOCUMENTS_JSONL_PATH
    json_path = config.PROCESSED_DOCUMENTS_JSON_PATH

    assert titles_path.exists()
    assert jsonl_path.exists(), "Run `python -m src.documents` first."
    assert json_path.exists(), "Run `python -m src.documents` first."

    titles = json.loads(Path(titles_path).read_text(encoding="utf-8"))
    docs_json = json.loads(Path(json_path).read_text(encoding="utf-8"))
    docs_jsonl = [
        json.loads(line)
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(docs_json) == 3000
    assert len(docs_jsonl) == 3000
    assert docs_json == docs_jsonl

    titles_by_id = {row["title_id"]: row for row in titles}
    seen_ids: set[str] = set()
    for doc in docs_json:
        tid = doc["id"]
        assert tid not in seen_ids
        seen_ids.add(tid)
        assert doc["metadata"]["title_id"] == tid
        assert doc["text"] == titles_by_id[tid]["search_text"]
        assert len(doc["text"].split()) <= 350
        for key, val in doc["metadata"].items():
            assert isinstance(val, (str, int, float, bool)), (
                f"Non-scalar metadata in {tid}.{key}: {type(val)}"
            )
