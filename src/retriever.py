"""Phase 4 filter-first hybrid retrieval engine (SQLite + ChromaDB + FastEmbed)."""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import re
import sqlite3
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src import config
from src.index import get_chroma_client, get_embedder

CANONICAL_LANGUAGES: tuple[str, ...] = (
    "English",
    "Hindi",
    "Tamil",
    "Telugu",
    "Malayalam",
    "Kannada",
    "Korean",
    "Japanese",
    "Spanish",
    "French",
)

LANGUAGE_ALIASES: dict[str, str] = {
    "english": "English",
    "eng": "English",
    "inglish": "English",
    "engilsh": "English",
    "hindi": "Hindi",
    "hindhi": "Hindi",
    "hndi": "Hindi",
    "bollywood": "Hindi",
    "tamil": "Tamil",
    "tamizh": "Tamil",
    "tmil": "Tamil",
    "kollywood": "Tamil",
    "telugu": "Telugu",
    "telegu": "Telugu",
    "telgu": "Telugu",
    "tollywood": "Telugu",
    "malayalam": "Malayalam",
    "malyalam": "Malayalam",
    "malayalm": "Malayalam",
    "mollywood": "Malayalam",
    "kannada": "Kannada",
    "kanada": "Kannada",
    "kannad": "Kannada",
    "kannnada": "Kannada",
    "sandalwood": "Kannada",
    "korean": "Korean",
    "kroean": "Korean",
    "kdrama": "Korean",
    "k-drama": "Korean",
    "japanese": "Japanese",
    "japnese": "Japanese",
    "anime": "Japanese",
    "spanish": "Spanish",
    "spanich": "Spanish",
    "espanol": "Spanish",
    "french": "French",
    "frnch": "French",
}

PLATFORM_GROUP_ALIASES: dict[str, str] = {
    "netflix": "Netflix",
    "amazon prime video": "Amazon Prime Video",
    "amazon prime": "Amazon Prime Video",
    "prime video": "Amazon Prime Video",
    "prime": "Amazon Prime Video",
    "prime video add on": "Prime Video add on",
    "amazon rent or buy": "Amazon (rent or buy)",
    "amazon (rent or buy)": "Amazon (rent or buy)",
    "jiohotstar": "JioHotstar",
    "jio hotstar": "JioHotstar",
    "hotstar": "JioHotstar",
    "disney+ hotstar": "JioHotstar",
    "disney hotstar": "JioHotstar",
    "zee5": "ZEE5",
    "zee 5": "ZEE5",
    "sonyliv": "SonyLIV",
    "sony liv": "SonyLIV",
    "aha": "aha",
    "sun nxt": "Sun NXT",
    "sunnxt": "Sun NXT",
    "youtube": "YouTube",
    "yt": "YouTube",
    "mx player": "MX Player",
    "mxplayer": "MX Player",
    "apple tv": "Apple TV",
    "apple tv+": "Apple TV",
    "apple tv add on": "Apple TV add on",
    "google play movies": "Google Play Movies",
    "google play": "Google Play Movies",
    "crunchyroll": "Crunchyroll",
    "mubi": "MUBI",
    "lionsgate play": "Lionsgate Play",
    "manoramamax": "ManoramaMAX",
    "shemaroome": "ShemarooMe",
    "vi movies and tv": "Vi Movies and TV",
    "hungama play": "Hungama Play",
}

CANONICAL_GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Kids",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Sci-Fi & Fantasy",
    "Thriller",
    "War",
    "Western",
)

CONTENT_NOTE_KEYWORDS: dict[str, str] = {
    "gore": "Gore",
    "blood": "Gore",
    "violence": "Violence",
    "violent": "Violence",
    "sexual": "Sexual content",
    "sex": "Sexual content",
    "nudity": "Sexual content",
    "drugs": "Drugs",
    "drug": "Drugs",
    "suicide": "Suicide or self harm",
    "self harm": "Suicide or self harm",
    "self-harm": "Suicide or self harm",
    "adult": "Adults only (A)",
}

BAYESIAN_PRIOR_VOTES: float = 100.0
CATALOG_MEAN_RATING: float = 6.8


class QueryFilters(BaseModel):
    """Structured filters supported against SQLite titles and where_to_watch."""

    type: Literal["Movie", "Series"] | None = None
    languages: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    era: list[str] = Field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    max_runtime_min: int | None = None
    min_runtime_min: int | None = None
    length_buckets: list[str] = Field(default_factory=list)
    max_seasons: int | None = None
    max_binge_hours: float | None = None
    status: str | None = None
    platforms: list[str] = Field(default_factory=list)
    access_types: list[str] = Field(default_factory=list)
    free_to_watch: bool | None = None
    certifications: list[str] = Field(default_factory=list)
    exclude_certifications: list[str] = Field(default_factory=list)
    intensity: str | None = None
    watch_with: str | None = None
    genres: list[str] = Field(default_factory=list)
    moods: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    avoid_content_notes: list[str] = Field(default_factory=list)
    min_rating: float | None = None
    person: str | None = None

    def active_filter_dict(self) -> dict[str, Any]:
        """Return only non-empty active filters for clean logging/inspection."""
        return {
            k: v
            for k, v in self.model_dump().items()
            if v not in (None, [], "")
        }


class QuerySpec(BaseModel):
    """Parsed user intent combining mode, semantic query, anchor, and filters."""

    raw_query: str
    mode: Literal["vibe_search", "more_like", "filter_only"] = "vibe_search"
    semantic_query: str = ""
    anchor_title: str | None = None
    anchor_language: str | None = None
    anchor_year: int | None = None
    filters: QueryFilters = Field(default_factory=QueryFilters)
    sort_preference: Literal["relevance", "top_rated", "hidden_gem"] = "relevance"


class WatchOption(BaseModel):
    """Normalized Indian streaming/rental/purchase availability option."""

    platform: str
    access_type: str
    region: str
    as_of: str


class PickPayload(BaseModel):
    """Grounded catalog record returned by retrieval."""

    title_id: str
    type: str
    title: str
    original_title: str | None = None
    language: str | None = None
    industry: str | None = None
    release_year: int | None = None
    runtime_min: int | None = None
    seasons: int | None = None
    episodes: int | None = None
    certification_in: str = "Unknown"
    genres: str | None = None
    mood: str | None = None
    themes: str | None = None
    intensity: str | None = None
    watch_with: str | None = None
    content_notes: str | None = None
    hook: str | None = None
    overview: str | None = None
    director_or_creator: str | None = None
    top_cast: str | None = None
    rating: float | None = None
    vote_count: int | None = None
    popularity: float | None = None
    poster_url: str | None = None
    watch_page_url: str | None = None
    tmdb_url: str | None = None
    imdb_url: str | None = None
    where_to_watch: list[WatchOption] = Field(default_factory=list)
    similarity_score: float | None = None
    final_score: float = 0.0


class DisambiguationOption(BaseModel):
    """Candidate anchor title when multiple catalog titles share a name."""

    title_id: str
    title: str
    release_year: int | None = None
    language: str | None = None
    type: str


class RelaxationSuggestion(BaseModel):
    """Single-filter ablation suggestion when strict filters return 0 matches."""

    filter_key: str
    label: str
    unlocked_count: int


class RetrievalResult(BaseModel):
    """Structured output of the Phase 4 hybrid retrieval engine."""

    query_spec: QuerySpec
    status: Literal["ok", "no_matches", "clarification_needed"] = "ok"
    picks: list[PickPayload] = Field(default_factory=list)
    total_candidates: int = 0
    clarification_question: str | None = None
    disambiguation_options: list[DisambiguationOption] = Field(default_factory=list)
    relaxation_suggestions: list[RelaxationSuggestion] = Field(default_factory=list)
    anchor_used: DisambiguationOption | None = None
    latency_ms: float = 0.0


def normalize_language(token: str) -> str | None:
    """Resolve exact, alias, or typo-tolerant language name to canonical form."""
    cleaned = re.sub(r"[^a-zA-Z\-]", "", token).lower().strip()
    if not cleaned:
        return None
    if cleaned in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[cleaned]
    for lang in CANONICAL_LANGUAGES:
        if cleaned == lang.lower():
            return lang
    matches = difflib.get_close_matches(
        cleaned,
        [lang.lower() for lang in CANONICAL_LANGUAGES],
        n=1,
        cutoff=0.76,
    )
    if matches:
        matched_lower = matches[0]
        return next(lang for lang in CANONICAL_LANGUAGES if lang.lower() == matched_lower)
    return None


def normalize_platform(token: str) -> str | None:
    """Map platform names/aliases to canonical platform_group values."""
    cleaned = re.sub(r"\s+", " ", token.lower().strip())
    return PLATFORM_GROUP_ALIASES.get(cleaned)


def _extract_languages(text_lower: str) -> list[str]:
    """Extract canonical languages with typo tolerance from natural-language text."""
    found: list[str] = []
    # Check multi-word/alias keys first, then individual words with fuzzy matching.
    words = re.findall(r"[a-zA-Z\-]+", text_lower)
    stopwords = {
        "film", "films", "movie", "movies", "series", "show", "shows",
        "watch", "free", "light", "heavy", "medium", "under", "hours",
        "hour", "thriller", "comedy", "drama", "horror", "action",
        "romance", "family", "friends", "alone", "date", "something",
        "like", "more", "than", "with", "without", "avoid", "from",
        "that", "this", "good", "best", "feel", "cozy", "comfort",
        "mind", "bending", "sad", "love", "story", "india", "indian",
        "short", "long", "quick", "binge", "dark", "epic", "funny",
        "about", "great", "want", "find", "give", "show", "only",
        "kantara", "drishyam", "leo", "inception",
    }
    for word in words:
        if len(word) < 3 or word in stopwords:
            continue
        lang = normalize_language(word)
        if lang and lang not in found:
            found.append(lang)

    if "south indian" in text_lower:
        for lang in ("Tamil", "Telugu", "Malayalam", "Kannada"):
            if lang not in found:
                found.append(lang)
    return found


def _extract_platforms(text_lower: str) -> list[str]:
    """Extract canonical platform_group filters from user text."""
    found: list[str] = []
    for alias in sorted(PLATFORM_GROUP_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text_lower):
            canonical = PLATFORM_GROUP_ALIASES[alias]
            if canonical not in found:
                found.append(canonical)
    return found


def parse_query_rule_based(user_query: str) -> QuerySpec:
    """Deterministic query parser for Phase 4 retrieval and Phase 5 fallback."""
    raw = user_query.strip()
    lower = raw.lower()
    filters = QueryFilters()

    # 1. Detect "more like <title>" / "something like <title>" / "similar to <title>"
    more_like_match = re.search(
        r"(?:more\s+like|something\s+like|movies?\s+like|series\s+like|shows?\s+like|similar\s+to)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    mode: Literal["vibe_search", "more_like", "filter_only"] = "vibe_search"
    anchor_title: str | None = None
    anchor_language: str | None = None
    anchor_year: int | None = None

    if more_like_match:
        mode = "more_like"
        tail = more_like_match.group(1).strip().rstrip("?.!")
        # Split trailing filter clauses like "on netflix" or "in hindi" if present.
        tail = re.split(
            r"\b(?:on|under|free\s+to\s+watch|without|avoid)\b",
            tail,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        year_m = re.search(r"\b(19\d{2}|20\d{2})\b", tail)
        if year_m:
            anchor_year = int(year_m.group(1))
            tail = (tail[: year_m.start()] + tail[year_m.end() :]).strip(" ,()")

        # Detect language inside anchor phrase (e.g. "Malayalam Drishyam", "Drishyam Hindi")
        anchor_langs = _extract_languages(tail.lower())
        if anchor_langs:
            anchor_language = anchor_langs[0]
            # Remove language words from anchor_title so SQLite looks up "Drishyam" cleanly
            tokens = [
                t
                for t in tail.split()
                if normalize_language(t) != anchor_language
            ]
            tail = " ".join(tokens).strip()

        anchor_title = tail.strip('"\' ')

    # 2. Type
    if re.search(r"\b(series|tv\s*show|shows|web\s*series|season|seasons|episodes)\b", lower):
        filters.type = "Series"
    elif re.search(r"\b(movie|movies|film|films|cinema)\b", lower):
        filters.type = "Movie"

    # 3. Languages (outside anchor-only modification)
    if mode == "more_like":
        # Check if language was mentioned outside the anchor phrase (e.g., "Hindi movies like Leo")
        prefix = raw[: more_like_match.start()] if more_like_match else ""
        suffix_after_anchor = ""
        if more_like_match:
            m_tail = more_like_match.group(1)
            split_tail = re.split(
                r"\b(?:on|under|free\s+to\s+watch|without|avoid)\b",
                m_tail,
                maxsplit=1,
                flags=re.IGNORECASE,
            )
            if len(split_tail) > 1:
                suffix_after_anchor = split_tail[1]
        filters.languages = _extract_languages((prefix + " " + suffix_after_anchor).lower())
    else:
        filters.languages = _extract_languages(lower)

    # 4. Era & Year
    era_matches = re.findall(r"\b(19\d0|20\d0)s\b", lower)
    if era_matches:
        filters.era = [f"{e}s" for e in dict.fromkeys(era_matches)]
    elif re.search(r"\b90s\b", lower):
        filters.era = ["1990s"]
    elif re.search(r"\b80s\b", lower):
        filters.era = ["1980s"]

    # 5. Runtime / Length
    under_hrs = re.search(r"under\s+(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", lower)
    under_mins = re.search(r"under\s+(\d+)\s*(?:minutes?|mins?)", lower)
    if under_hrs:
        filters.max_runtime_min = int(float(under_hrs.group(1)) * 60) - 1
    elif under_mins:
        filters.max_runtime_min = int(under_mins.group(1)) - 1
    elif re.search(r"\bshort\s+(?:movie|film|movies|films)\b", lower):
        filters.max_runtime_min = 95

    # 6. Platforms & Free to watch
    filters.platforms = _extract_platforms(lower)
    if re.search(r"\bfree(?:\s+to\s+watch|\s+with\s+ads)?\b", lower):
        filters.free_to_watch = True

    # 7. Intensity
    if re.search(r"\b(light|lighthearted|easy\s*watch)\b", lower):
        filters.intensity = "Light"
    elif re.search(r"\bheavy\b", lower):
        filters.intensity = "Heavy"

    # 8. Watch with & certificate safety (never drop Unknown by default)
    if re.search(r"\b(family|parents|mom|dad|kids)\b", lower):
        filters.watch_with = "Family"
        filters.exclude_certifications = ["A"]
    elif re.search(r"\bdate\b", lower):
        filters.watch_with = "Date"
    elif re.search(r"\bfriends\b", lower):
        filters.watch_with = "Friends"
    elif re.search(r"\balone\b", lower):
        filters.watch_with = "Alone"

    if re.search(r"\bavoid\s+a\s*rated\b|\bno\s+adult\b", lower):
        if "A" not in filters.exclude_certifications:
            filters.exclude_certifications.append("A")

    # 9. Content notes to avoid
    avoid_match = re.search(r"\b(?:avoid|no|without)\s+([a-zA-Z\s,/-]+)", lower)
    if avoid_match:
        avoid_span = avoid_match.group(1)
        for kw, note in CONTENT_NOTE_KEYWORDS.items():
            if kw in avoid_span and note not in filters.avoid_content_notes:
                filters.avoid_content_notes.append(note)

    # 10. Genres
    for genre in CANONICAL_GENRES:
        if re.search(rf"\b{re.escape(genre.lower())}\b", lower):
            filters.genres.append(genre)
    if re.search(r"\bsci[\s-]?fi\b", lower) and "Science Fiction" not in filters.genres:
        filters.genres.append("Science Fiction")

    # 11. Sort preference
    sort_pref: Literal["relevance", "top_rated", "hidden_gem"] = "relevance"
    if re.search(r"\b(hidden\s+gems?|underrated)\b", lower):
        sort_pref = "hidden_gem"
    elif re.search(r"\b(top\s+rated|highest\s+rated|best\s+rated)\b", lower):
        sort_pref = "top_rated"

    # 12. Semantic query
    semantic_query = raw
    if mode == "vibe_search":
        # Check if query is purely structured filters with minimal vibe words
        if not filters.genres and filters.intensity == "Light" and "light" in lower:
            semantic_query = "lighthearted feel good comforting easy watch"

    return QuerySpec(
        raw_query=raw,
        mode=mode,
        semantic_query=semantic_query,
        anchor_title=anchor_title,
        anchor_language=anchor_language,
        anchor_year=anchor_year,
        filters=filters,
        sort_preference=sort_pref,
    )


_EMBEDDER_LOCK = threading.Lock()
_COLLECTION_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_embedder_singleton():
    return get_embedder()


def get_cached_embedder():
    """Lazily load FastEmbed ONNX model once per process (thread-safe)."""
    with _EMBEDDER_LOCK:
        return _load_embedder_singleton()


def is_embedder_loaded() -> bool:
    """Return True if the lazy FastEmbed singleton has been initialized."""
    return _load_embedder_singleton.cache_info().currsize > 0


@lru_cache(maxsize=1)
def _load_collection_singleton():
    client = get_chroma_client(config.CHROMA_DIR)
    return client.get_collection(name=config.CHROMA_COLLECTION)


def get_cached_collection():
    """Open persistent ChromaDB collection once per process (thread-safe)."""
    with _COLLECTION_LOCK:
        return _load_collection_singleton()


def _open_sqlite(db_path: Path | None = None) -> sqlite3.Connection:
    target = db_path or config.SQLITE_DB_PATH
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    return conn


def _build_filter_sql(filters: QueryFilters) -> list[tuple[str, str, list[Any], str]]:
    """Build modular (filter_key, label, params, sql_clause) predicates."""
    clauses: list[tuple[str, str, list[Any], str]] = []

    if filters.type:
        clauses.append(("type", f"type = {filters.type}", [filters.type], "t.type = ?"))

    if filters.languages:
        ph = ", ".join("?" for _ in filters.languages)
        clauses.append(
            (
                "languages",
                f"language in {filters.languages}",
                list(filters.languages),
                f"t.language IN ({ph})",
            )
        )

    if filters.industries:
        ph = ", ".join("?" for _ in filters.industries)
        clauses.append(
            (
                "industries",
                f"industry in {filters.industries}",
                list(filters.industries),
                f"t.industry IN ({ph})",
            )
        )

    if filters.era:
        ph = ", ".join("?" for _ in filters.era)
        clauses.append(
            ("era", f"era in {filters.era}", list(filters.era), f"t.era IN ({ph})")
        )

    if filters.year_min is not None:
        clauses.append(
            ("year_min", f"year >= {filters.year_min}", [filters.year_min], "t.release_year >= ?")
        )

    if filters.year_max is not None:
        clauses.append(
            ("year_max", f"year <= {filters.year_max}", [filters.year_max], "t.release_year <= ?")
        )

    if filters.max_runtime_min is not None:
        clauses.append(
            (
                "max_runtime_min",
                f"runtime <= {filters.max_runtime_min} min",
                [filters.max_runtime_min],
                "t.runtime_min IS NOT NULL AND t.runtime_min <= ?",
            )
        )

    if filters.min_runtime_min is not None:
        clauses.append(
            (
                "min_runtime_min",
                f"runtime >= {filters.min_runtime_min} min",
                [filters.min_runtime_min],
                "t.runtime_min IS NOT NULL AND t.runtime_min >= ?",
            )
        )

    if filters.length_buckets:
        ph = ", ".join("?" for _ in filters.length_buckets)
        clauses.append(
            (
                "length_buckets",
                f"length_bucket in {filters.length_buckets}",
                list(filters.length_buckets),
                f"t.length_bucket IN ({ph})",
            )
        )

    if filters.max_seasons is not None:
        clauses.append(
            (
                "max_seasons",
                f"seasons <= {filters.max_seasons}",
                [filters.max_seasons],
                "t.seasons IS NOT NULL AND t.seasons <= ?",
            )
        )

    if filters.max_binge_hours is not None:
        clauses.append(
            (
                "max_binge_hours",
                f"binge_hours <= {filters.max_binge_hours}",
                [filters.max_binge_hours],
                "t.binge_hours IS NOT NULL AND t.binge_hours <= ?",
            )
        )

    if filters.status:
        clauses.append(("status", f"status = {filters.status}", [filters.status], "t.status = ?"))

    if filters.intensity:
        clauses.append(
            ("intensity", f"intensity = {filters.intensity}", [filters.intensity], "t.intensity = ?")
        )

    if filters.free_to_watch is True:
        clauses.append(("free_to_watch", "free_to_watch = True", [], "t.free_to_watch = 1"))

    if filters.certifications:
        # Never drop empty/Unknown certificates unless explicitly disallowed; include Unknown when no strict exclusion.
        certs = list(filters.certifications)
        ph = ", ".join("?" for _ in certs)
        clauses.append(
            (
                "certifications",
                f"certification in {certs}",
                certs,
                f"t.certification_in IN ({ph})",
            )
        )

    if filters.exclude_certifications:
        # Unknown certificates remain eligible (never dropped solely for missing certificate).
        ph = ", ".join("?" for _ in filters.exclude_certifications)
        clauses.append(
            (
                "exclude_certifications",
                f"exclude cert {filters.exclude_certifications}",
                list(filters.exclude_certifications),
                f"t.certification_in NOT IN ({ph})",
            )
        )

    if filters.min_rating is not None:
        clauses.append(
            (
                "min_rating",
                f"rating >= {filters.min_rating}",
                [filters.min_rating],
                "t.rating IS NOT NULL AND t.rating >= ?",
            )
        )

    if filters.watch_with:
        clauses.append(
            (
                "watch_with",
                f"watch_with = {filters.watch_with}",
                [filters.watch_with],
                "EXISTS (SELECT 1 FROM title_tags tg WHERE tg.title_id = t.title_id "
                "AND tg.tag_type = 'watch_with' AND tg.tag_value IN (?, 'Anyone'))",
            )
        )

    if filters.genres:
        ph = ", ".join("?" for _ in filters.genres)
        clauses.append(
            (
                "genres",
                f"genres match {filters.genres}",
                list(filters.genres),
                f"EXISTS (SELECT 1 FROM title_tags tg WHERE tg.title_id = t.title_id "
                f"AND tg.tag_type = 'genres' AND tg.tag_value IN ({ph}))",
            )
        )

    if filters.moods:
        ph = ", ".join("?" for _ in filters.moods)
        clauses.append(
            (
                "moods",
                f"moods match {filters.moods}",
                list(filters.moods),
                f"EXISTS (SELECT 1 FROM title_tags tg WHERE tg.title_id = t.title_id "
                f"AND tg.tag_type = 'mood' AND tg.tag_value IN ({ph}))",
            )
        )

    if filters.themes:
        ph = ", ".join("?" for _ in filters.themes)
        clauses.append(
            (
                "themes",
                f"themes match {filters.themes}",
                list(filters.themes),
                f"EXISTS (SELECT 1 FROM title_tags tg WHERE tg.title_id = t.title_id "
                f"AND tg.tag_type = 'themes' AND tg.tag_value IN ({ph}))",
            )
        )

    if filters.avoid_content_notes:
        ph = ", ".join("?" for _ in filters.avoid_content_notes)
        clauses.append(
            (
                "avoid_content_notes",
                f"avoid {filters.avoid_content_notes}",
                list(filters.avoid_content_notes),
                f"NOT EXISTS (SELECT 1 FROM title_tags tg WHERE tg.title_id = t.title_id "
                f"AND tg.tag_type = 'content_notes' AND tg.tag_value IN ({ph}))",
            )
        )

    if filters.person:
        like_p = f"%{filters.person}%"
        clauses.append(
            (
                "person",
                f"person = {filters.person}",
                [like_p, like_p],
                "(t.director_or_creator LIKE ? OR t.top_cast LIKE ?)",
            )
        )

    if filters.platforms and filters.access_types:
        ph_p = ", ".join("?" for _ in filters.platforms)
        ph_a = ", ".join("?" for _ in filters.access_types)
        clauses.append(
            (
                "platforms_access",
                f"platform in {filters.platforms} & access in {filters.access_types}",
                list(filters.platforms) + list(filters.access_types),
                f"EXISTS (SELECT 1 FROM where_to_watch w WHERE w.title_id = t.title_id "
                f"AND w.platform IN ({ph_p}) AND w.access_type IN ({ph_a}))",
            )
        )
    elif filters.platforms:
        ph_p = ", ".join("?" for _ in filters.platforms)
        clauses.append(
            (
                "platforms",
                f"platform_group in {filters.platforms}",
                list(filters.platforms),
                f"EXISTS (SELECT 1 FROM where_to_watch w WHERE w.title_id = t.title_id "
                f"AND w.platform IN ({ph_p}))",
            )
        )
    elif filters.access_types:
        ph_a = ", ".join("?" for _ in filters.access_types)
        clauses.append(
            (
                "access_types",
                f"access_type in {filters.access_types}",
                list(filters.access_types),
                f"EXISTS (SELECT 1 FROM where_to_watch w WHERE w.title_id = t.title_id "
                f"AND w.access_type IN ({ph_a}))",
            )
        )

    return clauses


def _run_candidate_query(
    conn: sqlite3.Connection, clauses: list[tuple[str, str, list[Any], str]]
) -> list[sqlite3.Row]:
    sql = "SELECT t.title_id, t.rating, t.vote_count, t.popularity FROM titles t"
    params: list[Any] = []
    if clauses:
        sql += " WHERE " + " AND ".join(c[3] for c in clauses)
        for c in clauses:
            params.extend(c[2])
    return conn.execute(sql, params).fetchall()


def _analyze_relaxations(
    conn: sqlite3.Connection, clauses: list[tuple[str, str, list[Any], str]]
) -> list[RelaxationSuggestion]:
    """Run progressive filter ablation in SQLite to suggest minimal relaxations."""
    n = len(clauses)
    if n == 0:
        return []

    # Try dropping k = 1, 2, ... filters until at least one relaxation yields matches.
    for num_drop in range(1, min(n, 6) + 1):
        suggestions: list[RelaxationSuggestion] = []
        for drop_indices in itertools.combinations(range(n), num_drop):
            drop_set = set(drop_indices)
            subset = [c for j, c in enumerate(clauses) if j not in drop_set]
            rows = _run_candidate_query(conn, subset)
            if rows:
                dropped_keys = ", ".join(clauses[j][0] for j in drop_indices)
                dropped_labels = "; ".join(clauses[j][1] for j in drop_indices)
                suggestions.append(
                    RelaxationSuggestion(
                        filter_key=dropped_keys,
                        label=f"Relax {dropped_labels}",
                        unlocked_count=len(rows),
                    )
                )
        if suggestions:
            suggestions.sort(key=lambda s: s.unlocked_count, reverse=True)
            return suggestions[:5]
    return []


def _normalize_title_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def resolve_anchor_candidates(
    conn: sqlite3.Connection,
    anchor_title: str,
) -> list[DisambiguationOption]:
    """Find catalog titles matching anchor_title (exact first, then normalized/prefix)."""
    # 1. Exact case-insensitive title or original_title match
    rows = conn.execute(
        """
        SELECT title_id, title, release_year, language, type
        FROM titles
        WHERE LOWER(TRIM(title)) = LOWER(TRIM(?))
           OR LOWER(TRIM(original_title)) = LOWER(TRIM(?))
        ORDER BY vote_count DESC
        """,
        (anchor_title, anchor_title),
    ).fetchall()

    # 2. Punctuation-normalized match (e.g. "spiderman" vs "Spider-Man")
    if not rows:
        target_norm = _normalize_title_key(anchor_title)
        all_rows = conn.execute(
            "SELECT title_id, title, original_title, release_year, language, type, vote_count "
            "FROM titles ORDER BY vote_count DESC"
        ).fetchall()
        rows = [
            r
            for r in all_rows
            if _normalize_title_key(r["title"]) == target_norm
            or (r["original_title"] and _normalize_title_key(r["original_title"]) == target_norm)
        ]

    # 3. FTS5 / substring fallback (e.g. "Kantara" or "3 idiot")
    if not rows:
        rows = conn.execute(
            """
            SELECT title_id, title, release_year, language, type
            FROM titles
            WHERE LOWER(title) LIKE LOWER(?)
            ORDER BY vote_count DESC
            LIMIT 5
            """,
            (f"%{anchor_title.strip()}%",),
        ).fetchall()

    return [
        DisambiguationOption(
            title_id=r["title_id"],
            title=r["title"],
            release_year=r["release_year"],
            language=r["language"],
            type=r["type"],
        )
        for r in rows
    ]


def _bayesian_score(rating: float | None, vote_count: int | None) -> float:
    """Compute Bayesian quality score normalized to [0, 1]."""
    r = float(rating) if rating is not None else 6.0
    v = float(vote_count) if vote_count is not None else 0.0
    bayes = (v / (v + BAYESIAN_PRIOR_VOTES)) * r + (
        BAYESIAN_PRIOR_VOTES / (v + BAYESIAN_PRIOR_VOTES)
    ) * CATALOG_MEAN_RATING
    return max(0.0, min(1.0, bayes / 10.0))


def _vector_rank_candidates(
    query_embedding: list[float],
    candidate_map: dict[str, sqlite3.Row],
    collection: Any,
    sort_preference: str,
    exclude_id: str | None = None,
    max_picks: int = config.MAX_PICKS,
) -> list[tuple[str, float, float]]:
    """Query ChromaDB and rank surviving candidate_ids by blended score."""
    candidate_ids = [cid for cid in candidate_map.keys() if cid != exclude_id]
    if not candidate_ids:
        return []

    sim_by_id: dict[str, float] = {}
    if len(candidate_ids) <= 200:
        res = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(len(candidate_ids), 50),
            where={"title_id": {"$in": candidate_ids}},
            include=["distances"],
        )
        for tid, dist in zip(res["ids"][0], res["distances"][0]):
            if tid != exclude_id:
                sim_by_id[tid] = max(0.0, min(1.0, 1.0 - float(dist)))
    else:
        # Broad vector query intersected with SQLite candidates to avoid giant $in lists
        total_col = collection.count()
        for n_fetch in (250, 750, total_col):
            res = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_fetch, total_col),
                include=["distances"],
            )
            for tid, dist in zip(res["ids"][0], res["distances"][0]):
                if tid in candidate_map and tid != exclude_id:
                    sim_by_id[tid] = max(0.0, min(1.0, 1.0 - float(dist)))
            if len(sim_by_id) >= max_picks * 3 or n_fetch >= total_col:
                break

    scored: list[tuple[str, float, float]] = []
    for tid, sim in sim_by_id.items():
        row = candidate_map[tid]
        qual = _bayesian_score(row["rating"], row["vote_count"])
        pop = float(row["popularity"] or 0.0)
        if sort_preference == "hidden_gem":
            pop_bonus = 0.08 if (pop < 25.0 and (row["rating"] or 0.0) >= 7.0) else 0.0
            final = 0.65 * sim + 0.35 * qual + pop_bonus
        elif sort_preference == "top_rated":
            final = 0.45 * sim + 0.55 * qual
        else:
            final = 0.75 * sim + 0.25 * qual
        scored.append((tid, round(sim, 4), round(final, 4)))

    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:max_picks]


def _hydrate_picks(
    conn: sqlite3.Connection,
    ranked_items: list[tuple[str, float | None, float]],
) -> list[PickPayload]:
    """Hydrate complete grounded PickPayloads from SQLite titles and where_to_watch."""
    picks: list[PickPayload] = []
    for tid, sim, final_score in ranked_items:
        row = conn.execute("SELECT * FROM titles WHERE title_id = ?", (tid,)).fetchone()
        if not row:
            continue
        wtw_rows = conn.execute(
            "SELECT platform, access_type, region, as_of FROM where_to_watch WHERE title_id = ?",
            (tid,),
        ).fetchall()
        watch_options = [
            WatchOption(
                platform=w["platform"],
                access_type=w["access_type"],
                region=w["region"],
                as_of=w["as_of"],
            )
            for w in wtw_rows
        ]
        picks.append(
            PickPayload(
                title_id=row["title_id"],
                type=row["type"],
                title=row["title"],
                original_title=row["original_title"],
                language=row["language"],
                industry=row["industry"],
                release_year=row["release_year"],
                runtime_min=row["runtime_min"],
                seasons=row["seasons"],
                episodes=row["episodes"],
                certification_in=row["certification_in"],
                genres=row["genres"],
                mood=row["mood"],
                themes=row["themes"],
                intensity=row["intensity"],
                watch_with=row["watch_with"],
                content_notes=row["content_notes"],
                hook=row["hook"],
                overview=row["overview"],
                director_or_creator=row["director_or_creator"],
                top_cast=row["top_cast"],
                rating=row["rating"],
                vote_count=row["vote_count"],
                popularity=row["popularity"],
                poster_url=row["poster_url"],
                watch_page_url=row["watch_page_url"],
                tmdb_url=row["tmdb_url"],
                imdb_url=row["imdb_url"],
                where_to_watch=watch_options,
                similarity_score=sim,
                final_score=final_score,
            )
        )
    return picks


def retrieve(
    query_or_spec: str | QuerySpec,
    db_path: Path | None = None,
    embedder: Any = None,
    collection: Any = None,
    max_picks: int = config.MAX_PICKS,
) -> RetrievalResult:
    """Execute Phase 4 hybrid retrieval from a user query string or QuerySpec."""
    t0 = time.perf_counter()
    spec = (
        parse_query_rule_based(query_or_spec)
        if isinstance(query_or_spec, str)
        else query_or_spec
    )
    col = collection or get_cached_collection()
    conn = _open_sqlite(db_path)

    try:
        # 1. Handle 'more_like' anchor resolution & duplicate-title disambiguation
        anchor_used: DisambiguationOption | None = None
        anchor_embedding: list[float] | None = None

        if spec.mode == "more_like" and spec.anchor_title:
            matches = resolve_anchor_candidates(conn, spec.anchor_title)
            if len(matches) > 1:
                # Prefer exact title matches if substring fallback wasn't needed
                exact_norm = _normalize_title_key(spec.anchor_title)
                exact_matches = [
                    m for m in matches if _normalize_title_key(m.title) == exact_norm
                ]
                if exact_matches:
                    matches = exact_matches

            if len(matches) > 1:
                # Try language/year disambiguation from user query
                preferred_lang = spec.anchor_language or (
                    spec.filters.languages[0] if len(spec.filters.languages) == 1 else None
                )
                filtered_matches = matches
                if preferred_lang:
                    lang_sub = [
                        m for m in filtered_matches if m.language == preferred_lang
                    ]
                    if lang_sub:
                        filtered_matches = lang_sub
                if spec.anchor_year:
                    yr_sub = [
                        m for m in filtered_matches if m.release_year == spec.anchor_year
                    ]
                    if yr_sub:
                        filtered_matches = yr_sub

                if len(filtered_matches) == 1:
                    anchor_used = filtered_matches[0]
                elif len(filtered_matches) > 1:
                    options_str = ", ".join(
                        f"{m.title} ({m.release_year}, {m.language} {m.type})"
                        for m in filtered_matches
                    )
                    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                    return RetrievalResult(
                        query_spec=spec,
                        status="clarification_needed",
                        clarification_question=(
                            f"Multiple titles match '{spec.anchor_title}': {options_str}. "
                            "Which one did you mean?"
                        ),
                        disambiguation_options=filtered_matches,
                        latency_ms=elapsed_ms,
                    )
            elif len(matches) == 1:
                anchor_used = matches[0]

            if anchor_used is not None:
                got = col.get(ids=[anchor_used.title_id], include=["embeddings"])
                embs = got.get("embeddings")
                if embs is not None and len(embs) > 0:
                    vec = embs[0]
                    anchor_embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            else:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                return RetrievalResult(
                    query_spec=spec,
                    status="no_matches",
                    picks=[],
                    total_candidates=0,
                    anchor_used=None,
                    latency_ms=elapsed_ms,
                )

        # 2. Filter-first SQLite candidate query
        clauses = _build_filter_sql(spec.filters)
        candidate_rows = _run_candidate_query(conn, clauses)
        candidate_map = {r["title_id"]: r for r in candidate_rows}

        if not candidate_map:
            relaxations = _analyze_relaxations(conn, clauses)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return RetrievalResult(
                query_spec=spec,
                status="no_matches",
                total_candidates=0,
                relaxation_suggestions=relaxations,
                anchor_used=anchor_used,
                latency_ms=elapsed_ms,
            )

        # 3. Rank candidates
        if spec.mode == "more_like":
            if anchor_embedding is None:
                emb = embedder or get_cached_embedder()
                q_text = f"movies and series similar to {spec.anchor_title or spec.raw_query}"
                anchor_embedding = next(emb.embed([q_text])).tolist()
            ranked = _vector_rank_candidates(
                query_embedding=anchor_embedding,
                candidate_map=candidate_map,
                collection=col,
                sort_preference=spec.sort_preference,
                exclude_id=anchor_used.title_id if anchor_used else None,
                max_picks=max_picks,
            )
            ranked_tuples: list[tuple[str, float | None, float]] = [
                (tid, sim, final) for tid, sim, final in ranked
            ]
        elif spec.mode == "filter_only" or not spec.semantic_query.strip():
            scored_rows = []
            for tid, row in candidate_map.items():
                qual = _bayesian_score(row["rating"], row["vote_count"])
                pop = float(row["popularity"] or 0.0)
                bonus = (
                    0.08
                    if (
                        spec.sort_preference == "hidden_gem"
                        and pop < 25.0
                        and (row["rating"] or 0.0) >= 7.0
                    )
                    else 0.0
                )
                scored_rows.append((tid, None, round(qual + bonus, 4), pop))
            scored_rows.sort(key=lambda x: (x[2], x[3]), reverse=True)
            ranked_tuples = [(tid, sim, sc) for tid, sim, sc, _ in scored_rows[:max_picks]]
        else:
            emb = embedder or get_cached_embedder()
            q_vec = next(emb.embed([spec.semantic_query])).tolist()
            ranked = _vector_rank_candidates(
                query_embedding=q_vec,
                candidate_map=candidate_map,
                collection=col,
                sort_preference=spec.sort_preference,
                max_picks=max_picks,
            )
            ranked_tuples = [(tid, sim, final) for tid, sim, final in ranked]

        picks = _hydrate_picks(conn, ranked_tuples)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return RetrievalResult(
            query_spec=spec,
            status="ok" if picks else "no_matches",
            picks=picks,
            total_candidates=len(candidate_map),
            anchor_used=anchor_used,
            latency_ms=elapsed_ms,
        )
    finally:
        conn.close()


QA_TEST_QUERIES: tuple[str, ...] = (
    "light Kannada or Hindi film under 2 hours, free to watch",
    "kanada thriller on hotstar",
    "hindhi comedy on prime video",
    "something like Kantara",
    "more like Drishyam",
    "more like Malayalam Drishyam",
    "more like Leo",
    "1950s Korean horror series free on aha under 30 min",
)


def main() -> None:
    """Run Phase 4 QA queries and print parsed filters and results."""
    parser = argparse.ArgumentParser(description="Movie Buddy Phase 4 retrieval QA")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional single query to test instead of the QA suite",
    )
    args = parser.parse_args()

    queries = (args.query,) if args.query else QA_TEST_QUERIES
    # Warm singletons before timing individual QA queries
    get_cached_embedder()
    get_cached_collection()

    for q in queries:
        res = retrieve(q)
        print("=" * 80)
        print(f"QUERY: {q!r}")
        print(f"MODE:  {res.query_spec.mode} | ANCHOR: {res.query_spec.anchor_title!r} "
              f"(lang={res.query_spec.anchor_language}, used={res.anchor_used})")
        print(
            "PARSED FILTERS:",
            json.dumps(res.query_spec.filters.active_filter_dict(), ensure_ascii=False),
        )
        print(
            f"STATUS: {res.status} | CANDIDATES: {res.total_candidates} | LATENCY: {res.latency_ms} ms"
        )
        if res.status == "clarification_needed":
            print(f"CLARIFICATION: {res.clarification_question}")
        elif res.status == "no_matches":
            print("RELAXATIONS:", [s.model_dump() for s in res.relaxation_suggestions[:3]])
        else:
            for idx, p in enumerate(res.picks, 1):
                plats = sorted({f"{w.platform} ({w.access_type})" for w in p.where_to_watch})
                print(
                    f"  {idx}. {p.title_id} | {p.title} ({p.release_year}, {p.language}, {p.type}, "
                    f"cert={p.certification_in}, runtime={p.runtime_min}, rating={p.rating}) "
                    f"| sim={p.similarity_score} final={p.final_score}"
                )
                print(f"     watch: {plats}")


if __name__ == "__main__":
    main()
