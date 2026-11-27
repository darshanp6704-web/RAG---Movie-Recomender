"""Phase 1 data loading, cleaning, joining, and persistence pipeline."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl

from src import config

EXPECTED_TITLES_COLUMNS: tuple[str, ...] = (
    "title_id",
    "type",
    "title",
    "original_title",
    "language",
    "industry",
    "origin_country",
    "release_year",
    "era",
    "runtime_min",
    "length_bucket",
    "seasons",
    "episodes",
    "binge_hours",
    "status",
    "certification_in",
    "genres",
    "keywords",
    "mood",
    "themes",
    "intensity",
    "watch_with",
    "content_notes",
    "hook",
    "overview",
    "director_or_creator",
    "top_cast",
    "rating",
    "vote_count",
    "popularity",
    "streaming_in",
    "free_to_watch",
    "watch_page_url",
    "poster_url",
    "tmdb_url",
    "imdb_url",
    "has_plot",
    "search_text",
    "data_as_of",
)

EXPECTED_WTW_COLUMNS: tuple[str, ...] = (
    "title_id",
    "title",
    "platform",
    "platform_group",
    "access_type",
    "region",
    "as_of",
)

INT_FIELDS: frozenset[str] = frozenset(
    {"release_year", "runtime_min", "seasons", "episodes", "vote_count"}
)
FLOAT_FIELDS: frozenset[str] = frozenset({"binge_hours", "rating", "popularity"})
BOOL_FIELDS: frozenset[str] = frozenset({"free_to_watch", "has_plot"})
MULTI_VALUE_FIELDS: tuple[str, ...] = (
    "genres",
    "keywords",
    "mood",
    "themes",
    "watch_with",
    "content_notes",
    "streaming_in",
    "director_or_creator",
    "top_cast",
)


def clean_text(value: Any) -> str | None:
    """Trim whitespace and convert empty strings to None."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def clean_csv_text(value: Any) -> str | None:
    """Normalize comma-separated values by trimming each item and dropping empties."""
    raw = clean_text(value)
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return ", ".join(parts) if parts else None


def clean_certificate(value: Any) -> str:
    """Trim India certificate and map empty values to 'Unknown'."""
    cert = clean_text(value)
    return cert if cert is not None else "Unknown"


def to_int(value: Any) -> int | None:
    """Coerce value to int while preserving None for empty cells."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = clean_text(value)
    return int(float(text)) if text is not None else None


def to_float(value: Any, ndigits: int = 2) -> float | None:
    """Coerce value to rounded float while preserving None for empty cells."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), ndigits)
    text = clean_text(value)
    return round(float(text), ndigits) if text is not None else None


def to_bool(value: Any) -> bool:
    """Coerce Excel booleans, 0/1 ints, or strings to Python bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = clean_text(value)
    if text is None:
        return False
    return text.lower() in {"1", "true", "yes", "y"}


def split_tags(value: str | None) -> list[str]:
    """Split a cleaned comma-separated field into individual tag strings."""
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def clean_title_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Clean and type-coerce a single row from the Titles sheet."""
    cleaned: dict[str, Any] = {}
    for col in EXPECTED_TITLES_COLUMNS:
        val = raw.get(col)
        if col == "certification_in":
            cleaned[col] = clean_certificate(val)
        elif col in INT_FIELDS:
            cleaned[col] = to_int(val)
        elif col in FLOAT_FIELDS:
            cleaned[col] = to_float(val, ndigits=2)
        elif col in BOOL_FIELDS:
            cleaned[col] = to_bool(val)
        elif col in MULTI_VALUE_FIELDS:
            cleaned[col] = clean_csv_text(val)
        else:
            cleaned[col] = clean_text(val)
    return cleaned


def clean_watch_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Clean a single row from Where to watch, using platform_group for platform."""
    platform_name = clean_text(raw.get("platform_group")) or clean_text(
        raw.get("platform")
    )
    return {
        "title_id": clean_text(raw.get("title_id")),
        "title": clean_text(raw.get("title")),
        "platform": platform_name,
        "access_type": clean_text(raw.get("access_type")),
        "region": clean_text(raw.get("region")) or "IN",
        "as_of": clean_text(raw.get("as_of")),
    }


def _read_sheet_dicts(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    expected_cols: tuple[str, ...],
    sheet_name: str,
) -> list[dict[str, Any]]:
    """Read an openpyxl worksheet into a list of dicts and validate headers."""
    rows_iter = ws.iter_rows(values_only=True)
    header_row = next(rows_iter, None)
    if header_row is None:
        raise ValueError(f"Sheet '{sheet_name}' is empty.")

    headers = [str(col).strip() if col is not None else "" for col in header_row]
    missing = [col for col in expected_cols if col not in headers]
    if missing:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing expected columns: {missing}"
        )

    records: list[dict[str, Any]] = []
    for row in rows_iter:
        if row is None or not any(cell is not None for cell in row):
            continue
        record = {
            headers[idx]: row[idx] if idx < len(row) else None
            for idx in range(len(headers))
            if headers[idx]
        }
        records.append(record)
    return records


def load_and_clean_catalog(
    excel_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load, validate, clean, and join Titles and Where to watch sheets."""
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Source workbook not found at '{excel_path}'. "
            f"Expected workbook at '{config.DEFAULT_EXCEL_PATH}'."
        )

    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    try:
        for required_sheet in ("Titles", "Where to watch"):
            if required_sheet not in wb.sheetnames:
                raise ValueError(
                    f"Missing required sheet '{required_sheet}' in '{excel_path}'."
                )

        raw_titles = _read_sheet_dicts(wb["Titles"], EXPECTED_TITLES_COLUMNS, "Titles")
        raw_wtw = _read_sheet_dicts(
            wb["Where to watch"], EXPECTED_WTW_COLUMNS, "Where to watch"
        )
    finally:
        wb.close()

    cleaned_titles = [clean_title_row(row) for row in raw_titles]
    valid_title_ids = {row["title_id"] for row in cleaned_titles if row["title_id"]}

    cleaned_wtw: list[dict[str, Any]] = []
    wtw_by_title: dict[str, list[dict[str, Any]]] = {}
    seen_per_title: dict[str, set[tuple[str | None, str | None, str | None, str | None]]] = {}

    for raw in raw_wtw:
        entry = clean_watch_row(raw)
        tid = entry["title_id"]
        if not tid or tid not in valid_title_ids:
            raise ValueError(f"Orphan Where to watch title_id detected: {tid!r}")
        cleaned_wtw.append(entry)

        dedup_key = (
            entry["platform"],
            entry["access_type"],
            entry["region"],
            entry["as_of"],
        )
        seen = seen_per_title.setdefault(tid, set())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        wtw_by_title.setdefault(tid, []).append(
            {
                "platform": entry["platform"],
                "access_type": entry["access_type"],
                "region": entry["region"],
                "as_of": entry["as_of"],
            }
        )

    for title_row in cleaned_titles:
        title_row["where_to_watch"] = wtw_by_title.get(title_row["title_id"], [])

    return cleaned_titles, cleaned_wtw


def save_processed_titles(titles: list[dict[str, Any]], output_path: Path) -> Path:
    """Persist cleaned, joined title records to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(titles, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def save_sqlite_store(
    titles: list[dict[str, Any]],
    watch_rows: list[dict[str, Any]],
    db_path: Path,
) -> Path:
    """Persist cleaned Titles, Where to watch, tags, and FTS5 index to SQLite."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".db.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(tmp_path)
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE titles (
                title_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                original_title TEXT,
                language TEXT,
                industry TEXT,
                origin_country TEXT,
                release_year INTEGER,
                era TEXT,
                runtime_min INTEGER,
                length_bucket TEXT,
                seasons INTEGER,
                episodes INTEGER,
                binge_hours REAL,
                status TEXT,
                certification_in TEXT NOT NULL,
                genres TEXT,
                keywords TEXT,
                mood TEXT,
                themes TEXT,
                intensity TEXT,
                watch_with TEXT,
                content_notes TEXT,
                hook TEXT,
                overview TEXT,
                director_or_creator TEXT,
                top_cast TEXT,
                rating REAL,
                vote_count INTEGER,
                popularity REAL,
                streaming_in TEXT,
                free_to_watch INTEGER NOT NULL,
                watch_page_url TEXT,
                poster_url TEXT,
                tmdb_url TEXT,
                imdb_url TEXT,
                has_plot INTEGER NOT NULL,
                search_text TEXT,
                data_as_of TEXT
            );

            CREATE TABLE title_tags (
                title_id TEXT NOT NULL,
                tag_type TEXT NOT NULL,
                tag_value TEXT NOT NULL,
                FOREIGN KEY (title_id) REFERENCES titles(title_id)
            );

            CREATE TABLE where_to_watch (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title_id TEXT NOT NULL,
                title TEXT,
                platform TEXT NOT NULL,
                access_type TEXT NOT NULL,
                region TEXT NOT NULL,
                as_of TEXT NOT NULL,
                FOREIGN KEY (title_id) REFERENCES titles(title_id)
            );

            CREATE VIRTUAL TABLE titles_fts USING fts5(
                title_id UNINDEXED,
                title,
                original_title
            );
            """
        )

        title_cols = EXPECTED_TITLES_COLUMNS
        placeholders = ", ".join("?" for _ in title_cols)
        title_insert = f"INSERT INTO titles ({', '.join(title_cols)}) VALUES ({placeholders})"

        title_tuples = []
        tag_tuples = []
        fts_tuples = []
        for row in titles:
            vals = []
            for col in title_cols:
                val = row[col]
                if col in BOOL_FIELDS:
                    val = int(bool(val))
                vals.append(val)
            title_tuples.append(tuple(vals))

            tid = row["title_id"]
            fts_tuples.append((tid, row["title"] or "", row["original_title"] or ""))
            for tag_col in (
                "genres",
                "mood",
                "themes",
                "watch_with",
                "content_notes",
                "director_or_creator",
                "top_cast",
            ):
                for tag_val in split_tags(row[tag_col]):
                    tag_tuples.append((tid, tag_col, tag_val))

        cur.executemany(title_insert, title_tuples)
        cur.executemany(
            "INSERT INTO title_tags (title_id, tag_type, tag_value) VALUES (?, ?, ?)",
            tag_tuples,
        )
        cur.executemany(
            "INSERT INTO titles_fts (title_id, title, original_title) VALUES (?, ?, ?)",
            fts_tuples,
        )

        # Store deduplicated platform_group watch options per title for fast queries.
        wtw_tuples = []
        for row in titles:
            for opt in row["where_to_watch"]:
                wtw_tuples.append(
                    (
                        row["title_id"],
                        row["title"],
                        opt["platform"],
                        opt["access_type"],
                        opt["region"],
                        opt["as_of"],
                    )
                )
        cur.executemany(
            "INSERT INTO where_to_watch (title_id, title, platform, access_type, region, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            wtw_tuples,
        )

        cur.executescript(
            """
            CREATE INDEX idx_titles_type ON titles(type);
            CREATE INDEX idx_titles_language ON titles(language);
            CREATE INDEX idx_titles_industry ON titles(industry);
            CREATE INDEX idx_titles_era ON titles(era);
            CREATE INDEX idx_titles_year ON titles(release_year);
            CREATE INDEX idx_titles_cert ON titles(certification_in);
            CREATE INDEX idx_titles_intensity ON titles(intensity);
            CREATE INDEX idx_titles_free ON titles(free_to_watch);
            CREATE INDEX idx_titles_rating ON titles(rating);
            CREATE INDEX idx_tags_lookup ON title_tags(tag_type, tag_value, title_id);
            CREATE INDEX idx_wtw_title ON where_to_watch(title_id);
            CREATE INDEX idx_wtw_platform_access ON where_to_watch(platform, access_type);
            """
        )
        conn.commit()
    finally:
        conn.close()

    tmp_path.replace(db_path)
    return db_path


def summarize_catalog(
    titles: list[dict[str, Any]], raw_wtw_count: int
) -> dict[str, Any]:
    """Compute catalog counts by language, type, and sample rows."""
    lang_counts = dict(Counter(row["language"] for row in titles).most_common())
    type_counts = dict(Counter(row["type"] for row in titles).most_common())
    return {
        "title_count": len(titles),
        "raw_where_to_watch_count": raw_wtw_count,
        "joined_where_to_watch_count": sum(len(r["where_to_watch"]) for r in titles),
        "counts_by_type": type_counts,
        "counts_by_language": lang_counts,
        "samples": titles[:3],
    }


def run_phase1(
    excel_path: Path | None = None,
    json_path: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Execute Phase 1 end-to-end and return summary statistics."""
    src_excel = excel_path or config.EXCEL_PATH
    out_json = json_path or config.PROCESSED_TITLES_PATH
    out_db = db_path or config.SQLITE_DB_PATH

    titles, raw_wtw = load_and_clean_catalog(src_excel)
    save_processed_titles(titles, out_json)
    save_sqlite_store(titles, raw_wtw, out_db)
    return summarize_catalog(titles, len(raw_wtw))


def main() -> None:
    """CLI entry point for Phase 1 data cleaning and persistence."""
    parser = argparse.ArgumentParser(description="Movie Buddy Phase 1 data ingestion")
    parser.add_argument(
        "--excel",
        type=Path,
        default=config.EXCEL_PATH,
        help="Path to source Excel workbook",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=config.PROCESSED_TITLES_PATH,
        help="Output path for cleaned titles JSON",
    )
    parser.add_argument(
        "--db-out",
        type=Path,
        default=config.SQLITE_DB_PATH,
        help="Output path for SQLite database",
    )
    args = parser.parse_args()

    summary = run_phase1(
        excel_path=args.excel,
        json_path=args.json_out,
        db_path=args.db_out,
    )

    print(f"Total Titles: {summary['title_count']}")
    print(f"Raw Where to watch rows: {summary['raw_where_to_watch_count']}")
    print(
        f"Deduplicated joined Where to watch entries: {summary['joined_where_to_watch_count']}"
    )
    print("\nCounts by type:")
    for t_name, count in summary["counts_by_type"].items():
        print(f"  {t_name}: {count}")

    print("\nCounts by language:")
    for lang, count in summary["counts_by_language"].items():
        print(f"  {lang}: {count}")

    print("\n3 Sample rows:")
    print(json.dumps(summary["samples"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
