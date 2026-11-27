"""Phase 2 document construction and chunking audit pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src import config

# Scalar fields stored alongside each ChromaDB vector (None values omitted).
SCALAR_METADATA_FIELDS: tuple[str, ...] = (
    "title_id",
    "type",
    "language",
    "industry",
    "origin_country",
    "release_year",
    "era",
    "runtime_min",
    "length_bucket",
    "seasons",
    "certification_in",
    "intensity",
    "free_to_watch",
    "rating",
    "vote_count",
    "popularity",
    "has_plot",
)


def build_scalar_metadata(row: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Extract non-null ChromaDB-compatible scalar metadata from a title record."""
    metadata: dict[str, str | int | float | bool] = {}
    for field in SCALAR_METADATA_FIELDS:
        val = row.get(field)
        if val is None:
            continue
        if isinstance(val, (bool, int, float, str)):
            metadata[field] = val
    return metadata


def build_documents(
    titles: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build 1-title-per-document records and compute a length audit summary."""
    documents: list[dict[str, Any]] = []
    skipped_ids: list[str] = []
    word_counts: list[int] = []
    char_counts: list[int] = []

    for row in titles:
        title_id = row.get("title_id")
        has_plot = bool(row.get("has_plot"))
        search_text = row.get("search_text")

        if not title_id or not has_plot or not isinstance(search_text, str) or not search_text:
            skipped_ids.append(str(title_id))
            continue

        documents.append(
            {
                "id": title_id,
                "text": search_text,
                "metadata": build_scalar_metadata(row),
            }
        )
        word_counts.append(len(search_text.split()))
        char_counts.append(len(search_text))

    words_sorted = sorted(word_counts)
    chars_sorted = sorted(char_counts)
    count = len(documents)

    audit = {
        "document_count": count,
        "skipped_count": len(skipped_ids),
        "skipped_ids": skipped_ids,
        "words": {
            "min": words_sorted[0] if count else 0,
            "p50": words_sorted[count // 2] if count else 0,
            "p95": words_sorted[int(count * 0.95)] if count else 0,
            "max": words_sorted[-1] if count else 0,
        },
        "chars": {
            "min": chars_sorted[0] if count else 0,
            "p50": chars_sorted[count // 2] if count else 0,
            "p95": chars_sorted[int(count * 0.95)] if count else 0,
            "max": chars_sorted[-1] if count else 0,
        },
    }
    return documents, audit


def save_documents(
    documents: list[dict[str, Any]],
    jsonl_path: Path,
    json_path: Path,
) -> tuple[Path, Path]:
    """Write documents to both JSONL and pretty-printed JSON files."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    json_path.write_text(
        json.dumps(documents, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return jsonl_path, json_path


def run_phase2(
    titles_path: Path | None = None,
    jsonl_path: Path | None = None,
    json_path: Path | None = None,
) -> dict[str, Any]:
    """Load titles.json, build atomic documents, and persist JSONL + JSON outputs."""
    src_titles = titles_path or config.PROCESSED_TITLES_PATH
    out_jsonl = jsonl_path or config.PROCESSED_DOCUMENTS_JSONL_PATH
    out_json = json_path or config.PROCESSED_DOCUMENTS_JSON_PATH

    if not src_titles.exists():
        raise FileNotFoundError(
            f"Processed titles not found at '{src_titles}'. Run Phase 1 (`python -m src.ingest`) first."
        )

    titles = json.loads(src_titles.read_text(encoding="utf-8"))
    documents, audit = build_documents(titles)
    save_documents(documents, out_jsonl, out_json)
    return {"audit": audit, "jsonl_path": str(out_jsonl), "json_path": str(out_json)}


def main() -> None:
    """CLI entry point for Phase 2 document generation."""
    parser = argparse.ArgumentParser(description="Movie Buddy Phase 2 document builder")
    parser.add_argument(
        "--titles",
        type=Path,
        default=config.PROCESSED_TITLES_PATH,
        help="Path to data/processed/titles.json",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=config.PROCESSED_DOCUMENTS_JSONL_PATH,
        help="Output path for documents.jsonl",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=config.PROCESSED_DOCUMENTS_JSON_PATH,
        help="Output path for pretty-printed documents.json",
    )
    args = parser.parse_args()

    result = run_phase2(
        titles_path=args.titles,
        jsonl_path=args.jsonl_out,
        json_path=args.json_out,
    )
    audit = result["audit"]
    print(f"Documents built: {audit['document_count']}")
    print(f"Skipped titles: {audit['skipped_count']}")
    print(f"Word count stats (min/p50/p95/max): {audit['words']}")
    print(f"Char count stats (min/p50/p95/max): {audit['chars']}")
    print(f"Saved JSONL: {result['jsonl_path']}")
    print(f"Saved JSON:  {result['json_path']}")


if __name__ == "__main__":
    main()
