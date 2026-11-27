"""Central configuration for Movie Buddy."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present (never committed to Git).
load_dotenv(BASE_DIR / ".env")

DATA_DIR: Path = BASE_DIR / "data"
DOCS_DIR: Path = BASE_DIR / "docs"

DEFAULT_EXCEL_PATH: Path = DATA_DIR / "movie_buddy_starter_3000.xlsx"
EXCEL_PATH: Path = Path(os.getenv("EXCEL_PATH", str(DEFAULT_EXCEL_PATH)))
if not EXCEL_PATH.is_absolute():
    EXCEL_PATH = BASE_DIR / EXCEL_PATH

SQLITE_DB_PATH: Path = Path(
    os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "movie_buddy.db"))
)
if not SQLITE_DB_PATH.is_absolute():
    SQLITE_DB_PATH = BASE_DIR / SQLITE_DB_PATH

CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = BASE_DIR / CHROMA_DIR

FASTEMBED_CACHE_DIR: Path = Path(
    os.getenv("FASTEMBED_CACHE_DIR", str(BASE_DIR / ".fastembed_cache"))
)
if not FASTEMBED_CACHE_DIR.is_absolute():
    FASTEMBED_CACHE_DIR = BASE_DIR / FASTEMBED_CACHE_DIR

PROCESSED_TITLES_PATH: Path = Path(
    os.getenv("PROCESSED_TITLES_PATH", str(DATA_DIR / "processed" / "titles.json"))
)
if not PROCESSED_TITLES_PATH.is_absolute():
    PROCESSED_TITLES_PATH = BASE_DIR / PROCESSED_TITLES_PATH

PROCESSED_DOCUMENTS_JSONL_PATH: Path = DATA_DIR / "processed" / "documents.jsonl"
PROCESSED_DOCUMENTS_JSON_PATH: Path = DATA_DIR / "processed" / "documents.json"

CHROMA_COLLECTION: str = "movie_buddy_titles"
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MAX_TOKENS: int = max(1024, int(os.getenv("GROQ_MAX_TOKENS", "2048")))

MAX_PICKS: int = 5
TMDB_JUSTWATCH_FOOTER: str = (
    "This product uses the TMDB API but is not endorsed or certified by TMDB. "
    "Streaming availability by JustWatch."
)
