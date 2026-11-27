"""Phase 6 input and output guardrails for Movie Buddy."""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src import config
from src.retriever import PickPayload, WatchOption

logger = logging.getLogger("movie_buddy.guardrails")

PIRACY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:torrent|torrents|magnet\s*link|pirate|piracy|pirated)\b", re.I),
    re.compile(r"\b(?:camrip|hdrip|dvdrip|webrip|bluray\s*rip|x264|x265)\b", re.I),
    re.compile(
        r"\b(?:telegram\s*(?:link|channel|group)|free\s*download|download\s*(?:free|link|movie|film|series|mkv|mp4))\b",
        re.I,
    ),
    re.compile(
        r"\b(?:123movies|vegamovies|tamilrockers|filmyzilla|bolly4u|fzmovies|hdhub4u|soap2day|fmovies|moviesmod|mkvcinemas)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:without\s+paying|bypass\s+paywall|free\s+link\s+to\s+paid|watch\s+paid\s+.*\s+free|crack(?:ed)?\s+account)\b",
        re.I,
    ),
)

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|system|your)\s+(?:instructions|rules|prompt|constraints)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:reveal|print|show|dump|output|leak|repeat)\s+(?:your\s+|the\s+)?(?:system\s+prompt|developer\s+prompt|instructions|api\s*key|groq_api_key|\.env|secrets?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:even\s+if\s+not\s+in\s+(?:the\s+)?(?:excel|catalog|dataset)|invent\s+a\s+(?:movie|rating|platform)|pretend\s+you\s+are\s+unfiltered|jailbreak|dan\s+mode)\b",
        re.I,
    ),
    re.compile(r"\bGROQ_API_KEY\b"),
)

OFF_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:python|javascript|java|c\+\+|typescript|quicksort|mergesort|algorithm|sql\s+query|debug\s+(?:my\s+)?code|html|css|react)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:capital\s+of\s+[a-z]+|weather\s+(?:in|today|forecast)|stock\s+price|bitcoin|crypto|income\s+tax|taxes|calculus|derivative|integral|homework|essay)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:solve\s+\d+\s*[+\-*/^]\s*\d+|write\s+(?:a\s+)?(?:poem|email|cover\s+letter|code|script)|recipe\s+for|medical\s+advice|symptom|prescription)\b",
        re.I,
    ),
)

MOVIE_INTENT_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:movie|movies|film|films|series|tv\s*show|shows|web\s*series|cinema|watch|recommend|suggest|"
    r"thriller|comedy|drama|horror|action|romance|sci-fi|animation|documentary|bollywood|kollywood|"
    r"tollywood|mollywood|sandalwood|hindi|english|tamil|telugu|malayalam|kannada|korean|japanese|"
    r"netflix|prime|hotstar|jiohotstar|zee5|sonyliv|aha|youtube|mx\s*player|more\s+like|something\s+like)\b",
    re.I,
)

SPOILER_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:plot\s+twist|final\s+twist|ending\s+reveals|reveals\s+at\s+the\s+end)\b", re.I),
    re.compile(r"\b(?:dies\s+at\s+the\s+end|is\s+killed\s+in\s+the\s+end|turns\s+out\s+to\s+be\s+the\s+killer)\b", re.I),
    re.compile(r"\b(?:in\s+the\s+climax|spoiler\s+alert|major\s+spoiler)\b", re.I),
)

SPOILER_REQUEST_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:spoil|spoilers|tell\s+me\s+the\s+ending|ending\s+plot\s+twist|how\s+does\s+.*\s+end|who\s+is\s+the\s+killer|reveal\s+the\s+twist)\b",
    re.I,
)


class InputGuardrailResult(BaseModel):
    """Outcome of pre-LLM input guardrail evaluation."""

    allowed: bool = True
    violation_type: Literal["piracy", "prompt_injection", "off_topic", "spoiler"] | None = None
    refusal_message: str | None = None
    sanitized_query: str
    legal_picks: list[PickPayload] = Field(default_factory=list)


def _find_mentioned_catalog_titles(
    user_query: str,
    db_path: Path | None = None,
    limit: int = 3,
) -> list[PickPayload]:
    """Look up catalog titles mentioned in a piracy query so legal platforms can be offered."""
    target_db = db_path or config.SQLITE_DB_PATH
    if not target_db.exists():
        return []

    q_lower = user_query.lower()
    conn = sqlite3.connect(target_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT title_id, title, release_year, language, type, rating, certification_in, "
            "hook, overview, poster_url, watch_page_url, tmdb_url, imdb_url, runtime_min, seasons "
            "FROM titles ORDER BY LENGTH(title) DESC, vote_count DESC"
        ).fetchall()

        matched: list[PickPayload] = []
        seen_names: set[str] = set()
        for row in rows:
            t_name = (row["title"] or "").strip()
            if len(t_name) < 3:
                continue
            if re.search(rf"\b{re.escape(t_name.lower())}\b", q_lower):
                key = t_name.lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
                wtw = conn.execute(
                    "SELECT platform, access_type, region, as_of FROM where_to_watch WHERE title_id = ?",
                    (row["title_id"],),
                ).fetchall()
                matched.append(
                    PickPayload(
                        title_id=row["title_id"],
                        type=row["type"],
                        title=row["title"],
                        language=row["language"],
                        release_year=row["release_year"],
                        runtime_min=row["runtime_min"],
                        seasons=row["seasons"],
                        certification_in=row["certification_in"],
                        hook=row["hook"],
                        overview=row["overview"],
                        rating=row["rating"],
                        poster_url=row["poster_url"],
                        watch_page_url=row["watch_page_url"],
                        tmdb_url=row["tmdb_url"],
                        imdb_url=row["imdb_url"],
                        where_to_watch=[
                            WatchOption(
                                platform=w["platform"],
                                access_type=w["access_type"],
                                region=w["region"],
                                as_of=w["as_of"],
                            )
                            for w in wtw
                        ],
                    )
                )
                if len(matched) >= limit:
                    break
        return matched
    finally:
        conn.close()


def _strip_off_topic_clauses(user_query: str) -> str:
    """Remove off-topic leading clauses from mixed queries while preserving movie intent."""
    parts = re.split(r"\b(?:and\s+also|and\s+then|also|;|but)\b", user_query, flags=re.I)
    kept = [p.strip() for p in parts if MOVIE_INTENT_PATTERN.search(p)]
    return " ".join(kept).strip() if kept else user_query.strip()


def check_input_guardrails(
    user_query: str,
    db_path: Path | None = None,
) -> InputGuardrailResult:
    """Run anti-piracy, prompt-injection, and off-topic checks before calling Groq."""
    cleaned = user_query.strip()

    # 1. Anti-piracy check (always refuse and offer legal Indian platforms)
    if any(pat.search(cleaned) for pat in PIRACY_PATTERNS):
        legal_picks = _find_mentioned_catalog_titles(cleaned, db_path=db_path)
        if legal_picks:
            summaries = []
            for p in legal_picks:
                opts = ", ".join(
                    f"{w.platform} ({w.access_type}, as of {w.as_of})"
                    for w in p.where_to_watch
                )
                summaries.append(f"{p.title} ({p.release_year}): {opts}")
            msg = (
                "I can't help with torrents, downloads, or pirated links. "
                "Here are the legal ways to watch in India: " + " | ".join(summaries)
            )
        else:
            msg = (
                "I can't help with torrents, downloads, or pirated links. "
                "Instead, I can help you find legal Indian streaming or free-to-watch options "
                "on platforms like Netflix, Amazon Prime Video, JioHotstar, ZEE5, SonyLIV, MX Player, or YouTube."
            )
        logger.warning("Input blocked by piracy guardrail")
        return InputGuardrailResult(
            allowed=False,
            violation_type="piracy",
            refusal_message=msg,
            sanitized_query=cleaned,
            legal_picks=legal_picks,
        )

    # 2. Prompt injection / secret extraction check
    if any(pat.search(cleaned) for pat in INJECTION_PATTERNS):
        logger.warning("Input blocked by prompt-injection guardrail")
        return InputGuardrailResult(
            allowed=False,
            violation_type="prompt_injection",
            refusal_message=(
                "I follow strict Movie Buddy catalog rules and can only recommend verified movies "
                "and series from our Indian catalog. Tell me your mood, language, or platform preference!"
            ),
            sanitized_query=cleaned,
        )

    # 3. Explicit spoiler request check
    if SPOILER_REQUEST_PATTERN.search(cleaned):
        logger.warning("Input blocked by spoiler guardrail")
        return InputGuardrailResult(
            allowed=False,
            violation_type="spoiler",
            refusal_message=(
                "I keep all descriptions strictly spoiler-free using only official hooks and overviews, "
                "so I can't reveal endings or plot twists. Ask me for spoiler-free recommendations instead!"
            ),
            sanitized_query=cleaned,
        )

    # 4. Off-topic check (reject pure off-topic; isolate movie part in mixed queries)
    has_off_topic = any(pat.search(cleaned) for pat in OFF_TOPIC_PATTERNS)
    has_movie_intent = bool(MOVIE_INTENT_PATTERN.search(cleaned))

    if has_off_topic and not has_movie_intent:
        logger.warning("Input blocked by off-topic guardrail")
        return InputGuardrailResult(
            allowed=False,
            violation_type="off_topic",
            refusal_message=(
                "I'm Movie Buddy, so I only help with movie and series recommendations and where to "
                "legally watch them in India. Ask me for a genre, mood, language, or 'more like <title>'!"
            ),
            sanitized_query=cleaned,
        )

    if has_off_topic and has_movie_intent:
        sanitized = _strip_off_topic_clauses(cleaned)
        return InputGuardrailResult(
            allowed=True,
            sanitized_query=sanitized,
        )

    # Even without explicit off-topic keywords, reject clearly non-movie general trivia questions
    if re.match(r"^(?:who\s+is\s+the\s+president|what\s+is\s+\d+\s*[+\-*/]\s*\d+|define\s+)", cleaned, re.I) and not has_movie_intent:
        logger.warning("Input blocked by off-topic trivia guardrail")
        return InputGuardrailResult(
            allowed=False,
            violation_type="off_topic",
            refusal_message=(
                "I'm Movie Buddy, so I only help with movie and series recommendations in India."
            ),
            sanitized_query=cleaned,
        )

    return InputGuardrailResult(allowed=True, sanitized_query=cleaned)


def is_spoiler_free(text: str) -> bool:
    """Return False if text contains explicit plot-twist or ending spoiler phrasing."""
    if not text:
        return True
    return not any(pat.search(text) for pat in SPOILER_PHRASES)


def sanitize_spoiler_text(
    reason: str,
    fallback_hook: str | None,
    overview: str | None,
) -> str:
    """Ensure match description stays spoiler-free by falling back to safe hook/first overview sentence."""
    if is_spoiler_free(reason):
        return reason
    if fallback_hook and is_spoiler_free(fallback_hook):
        return fallback_hook
    if overview:
        first_sentence = overview.split(".")[0].strip()
        if first_sentence:
            return first_sentence + "."
    return "Matches your requested mood and genres without spoilers."


def enforce_output_guardrails(
    cards: list[Any],
    retrieved_picks: list[PickPayload],
    intro_message: str,
    user_query: str = "",
) -> tuple[list[Any], str]:
    """Drop unretrieved titles, attach SQLite watch options + as_of, and enforce spoiler-free text."""
    retrieved_by_id: dict[str, PickPayload] = {p.title_id: p for p in retrieved_picks}
    safe_cards: list[Any] = []

    for card in cards:
        src = retrieved_by_id.get(getattr(card, "title_id", ""))
        if src is None:
            logger.warning(
                "Output guardrail dropped unretrieved title_id=%r",
                getattr(card, "title_id", None),
            )
            continue

        # Enforce all factual fields & watch options directly from SQLite-hydrated PickPayload
        card.title = src.title
        card.release_year = src.release_year
        card.language = src.language
        card.type = src.type
        card.runtime_min = src.runtime_min
        card.seasons = src.seasons
        card.rating = src.rating
        card.certification_in = src.certification_in
        card.hook = src.hook
        card.tmdb_url = src.tmdb_url
        card.poster_url = src.poster_url
        card.watch_page_url = src.watch_page_url
        card.imdb_url = src.imdb_url

        # Always attach platforms & as_of from grounded data, never from LLM
        card.where_to_watch = [
            WatchOption(
                platform=w.platform,
                access_type=w.access_type,
                region=w.region,
                as_of=w.as_of,
            )
            for w in src.where_to_watch
            if w.platform and w.access_type and w.as_of
        ]

        # Enforce spoiler-free match reason
        card.why_it_matches = sanitize_spoiler_text(
            reason=card.why_it_matches,
            fallback_hook=src.hook,
            overview=src.overview,
        )
        safe_cards.append(card)
        if len(safe_cards) >= config.MAX_PICKS:
            break

    safe_intro = intro_message
    if not is_spoiler_free(safe_intro):
        safe_intro = "Here are spoiler-free picks from our Indian catalog that match your request."

    if SPOILER_REQUEST_PATTERN.search(user_query):
        safe_intro = (
            "I keep all descriptions spoiler-free using only official hooks and overviews. "
            + safe_intro
        )

    return safe_cards, safe_intro
