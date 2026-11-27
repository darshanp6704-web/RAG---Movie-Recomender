"""Phase 4 unit and integration tests for hybrid retrieval, disambiguation, and QA."""

from __future__ import annotations

import sqlite3

from src import config
from src.retriever import (
    QueryFilters,
    QuerySpec,
    normalize_language,
    parse_query_rule_based,
    retrieve,
)


def test_language_typo_tolerance_and_parsed_filters() -> None:
    """Verify typo tolerance for language names and platform_group parsing."""
    assert normalize_language("kanada") == "Kannada"
    assert normalize_language("hindhi") == "Hindi"
    assert normalize_language("malyalam") == "Malayalam"
    assert normalize_language("telegu") == "Telugu"

    spec1 = parse_query_rule_based("kanada thriller on hotstar")
    print("Parsed filters (kanada):", spec1.filters.active_filter_dict())
    assert spec1.filters.languages == ["Kannada"]
    assert spec1.filters.platforms == ["JioHotstar"]
    assert "Thriller" in spec1.filters.genres

    spec2 = parse_query_rule_based("hindhi comedy on prime video")
    print("Parsed filters (hindhi):", spec2.filters.active_filter_dict())
    assert spec2.filters.languages == ["Hindi"]
    assert spec2.filters.platforms == ["Amazon Prime Video"]
    assert "Comedy" in spec2.filters.genres


def test_light_kannada_or_hindi_under_2_hrs_free() -> None:
    """Verify exact filter-first hybrid retrieval for multi-constraint query."""
    q = "light Kannada or Hindi film under 2 hours, free to watch"
    res = retrieve(q)
    print("Parsed filters:", res.query_spec.filters.active_filter_dict())
    assert res.status == "ok"
    assert len(res.picks) == 2
    for pick in res.picks:
        assert pick.type == "Movie"
        assert pick.language in ("Kannada", "Hindi")
        assert pick.intensity == "Light"
        assert pick.runtime_min is not None and pick.runtime_min < 120
        assert any(w.access_type in ("Free", "Free with ads") for w in pick.where_to_watch)


def test_platform_group_filtering_prime_add_on_vs_subscription() -> None:
    """Verify platform filtering operates on platform_group values."""
    spec = QuerySpec(
        raw_query="Prime Video add on movies",
        mode="filter_only",
        filters=QueryFilters(type="Movie", platforms=["Prime Video add on"]),
    )
    print("Parsed filters (platform_group):", spec.filters.active_filter_dict())
    res = retrieve(spec)
    assert res.status == "ok"
    assert len(res.picks) > 0
    for pick in res.picks:
        assert any(w.platform == "Prime Video add on" for w in pick.where_to_watch)


def test_unknown_certificate_never_dropped() -> None:
    """Verify titles with empty/Unknown certification_in are never dropped solely for missing cert."""
    spec = QuerySpec(
        raw_query="family friendly Hindi movie",
        mode="filter_only",
        filters=QueryFilters(
            type="Movie",
            languages=["Hindi"],
            watch_with="Family",
            exclude_certifications=["A"],
        ),
    )
    print("Parsed filters (cert check):", spec.filters.active_filter_dict())
    # Confirm there are Hindi Family movies with Unknown cert and they remain eligible
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    unknown_count = conn.execute(
        """
        SELECT COUNT(*) FROM titles t
        WHERE t.type = 'Movie' AND t.language = 'Hindi'
          AND t.certification_in = 'Unknown'
          AND EXISTS (
              SELECT 1 FROM title_tags tg
              WHERE tg.title_id = t.title_id
                AND tg.tag_type = 'watch_with'
                AND tg.tag_value IN ('Family', 'Anyone')
          )
        """
    ).fetchone()[0]
    conn.close()
    assert unknown_count > 0

    res = retrieve(spec)
    assert res.status == "ok"
    assert res.total_candidates >= unknown_count


def test_more_like_uses_title_embedding_and_excludes_anchor() -> None:
    """Verify 'something like Kantara' uses Kantara's stored vector and excludes Kantara itself."""
    res = retrieve("something like Kantara")
    print("Parsed filters (more like Kantara):", res.query_spec.filters.active_filter_dict())
    assert res.status == "ok"
    assert res.anchor_used is not None
    assert res.anchor_used.title == "Kantara"
    assert all(p.title_id != res.anchor_used.title_id for p in res.picks)
    assert len(res.picks) == 5


def test_duplicate_title_disambiguation_drishyam_and_leo() -> None:
    """Verify shared-name disambiguation asks when language omitted, resolves when given."""
    # Unspecified language for Drishyam (exists in Hindi and Malayalam) -> ask which one
    res_ambig = retrieve("more like Drishyam")
    print("Parsed filters (Drishyam ambig):", res_ambig.query_spec.filters.active_filter_dict())
    assert res_ambig.status == "clarification_needed"
    langs = {opt.language for opt in res_ambig.disambiguation_options}
    assert {"Hindi", "Malayalam"}.issubset(langs)

    # Language mentioned -> resolves deterministically without asking
    res_ml = retrieve("more like Malayalam Drishyam")
    print("Parsed filters (Malayalam Drishyam):", res_ml.query_spec.filters.active_filter_dict())
    assert res_ml.status == "ok"
    assert res_ml.anchor_used is not None
    assert res_ml.anchor_used.language == "Malayalam"
    assert res_ml.anchor_used.title == "Drishyam"

    # Leo (single exact match in catalog: Tamil 2023) resolves directly
    res_leo = retrieve("more like Leo")
    print("Parsed filters (Leo):", res_leo.query_spec.filters.active_filter_dict())
    assert res_leo.status == "ok"
    assert res_leo.anchor_used is not None
    assert res_leo.anchor_used.title == "Leo"
    assert res_leo.anchor_used.language == "Tamil"


def test_over_constrained_returns_relaxation_suggestions() -> None:
    """Verify zero-match queries return actionable filter-relaxation suggestions."""
    res = retrieve("1950s Korean horror series free on aha under 30 min")
    print("Parsed filters (0 match):", res.query_spec.filters.active_filter_dict())
    assert res.status == "no_matches"
    assert res.picks == []
    assert len(res.relaxation_suggestions) > 0
