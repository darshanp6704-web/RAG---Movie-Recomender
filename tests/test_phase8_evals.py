"""Phase 8 evaluation runner executing all 20 cases in evals/questions.yaml."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from src import config
from src.groq_client import GroqServiceError
from src.guardrails import is_spoiler_free
from src.responder import recommend
from src.retriever import get_cached_collection, get_cached_embedder

EVALS_PATH: Path = config.BASE_DIR / "evals" / "questions.yaml"


def _load_catalog_lookup() -> dict[str, dict[str, Any]]:
    titles = json.loads(config.PROCESSED_TITLES_PATH.read_text(encoding="utf-8"))
    return {row["title_id"]: row for row in titles}


def _load_eval_cases() -> list[dict[str, Any]]:
    data = yaml.safe_load(EVALS_PATH.read_text(encoding="utf-8"))
    return data["cases"]


def validate_case(
    case: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> tuple[bool, float, str, int]:
    """Run one eval case end-to-end and return (passed, latency_s, detail, groq_calls)."""
    resp = None
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            resp = recommend(case["query"])
            latency_s = round(time.perf_counter() - t0, 2)
            break
        except GroqServiceError as exc:
            if exc.status_code == 429 and attempt < 2:
                # Pace batch evaluation within Groq's 8,000 TPM free-tier quota
                time.sleep(12)
                continue
            latency_s = round(time.perf_counter() - t0, 2)
            return False, latency_s, f"GroqServiceError: {exc.user_message}", 1
    assert resp is not None
    groq_calls = len(resp.groq_logs)

    if resp.status != case["expected_status"]:
        return (
            False,
            latency_s,
            f"status={resp.status!r}, expected={case['expected_status']!r}",
            groq_calls,
        )

    expected_violation = case.get("expected_violation")
    if expected_violation and resp.violation_type != expected_violation:
        return (
            False,
            latency_s,
            f"violation_type={resp.violation_type!r}, expected={expected_violation!r}",
            groq_calls,
        )

    if not (case["min_picks"] <= len(resp.picks) <= case["max_picks"]):
        return (
            False,
            latency_s,
            f"pick count={len(resp.picks)} outside [{case['min_picks']}, {case['max_picks']}]",
            groq_calls,
        )

    # Grounding & guardrail invariants on every returned pick
    for card in resp.picks:
        if card.title_id not in catalog:
            return False, latency_s, f"title_id {card.title_id!r} not in dataset", groq_calls
        truth = catalog[card.title_id]
        if card.title != truth["title"] or card.release_year != truth["release_year"]:
            return False, latency_s, f"metadata mismatch for {card.title_id}", groq_calls
        if not card.where_to_watch or not all(w.as_of for w in card.where_to_watch):
            return False, latency_s, f"missing watch/as_of on {card.title_id}", groq_calls
        if not is_spoiler_free(card.why_it_matches):
            return False, latency_s, f"spoiler phrasing detected on {card.title_id}", groq_calls

    # Expected filter assertions
    ef = case.get("expected_filters") or {}
    if ef.get("require_relaxation_suggestions") and not resp.relaxation_suggestions:
        return False, latency_s, "expected non-empty relaxation_suggestions", groq_calls

    msg_hints = ef.get("message_contains_any") or []
    if msg_hints and not any(h.lower() in resp.message.lower() for h in msg_hints):
        return False, latency_s, f"message missing expected terms {msg_hints}: {resp.message!r}", groq_calls

    for card in resp.picks:
        truth = catalog[card.title_id]
        if ef.get("type") and card.type != ef["type"]:
            return False, latency_s, f"{card.title_id} type={card.type} != {ef['type']}", groq_calls
        if ef.get("languages") and card.language not in ef["languages"]:
            return False, latency_s, f"{card.title_id} lang={card.language} not in {ef['languages']}", groq_calls
        if ef.get("intensity") and truth.get("intensity") != ef["intensity"]:
            return False, latency_s, f"{card.title_id} intensity={truth.get('intensity')} != {ef['intensity']}", groq_calls
        if ef.get("max_runtime_min") is not None:
            if card.runtime_min is None or card.runtime_min > ef["max_runtime_min"]:
                return False, latency_s, f"{card.title_id} runtime={card.runtime_min} > {ef['max_runtime_min']}", groq_calls
        if ef.get("free_to_watch") and not truth.get("free_to_watch"):
            return False, latency_s, f"{card.title_id} is not free_to_watch", groq_calls
        if ef.get("platform"):
            plats = {w.platform for w in card.where_to_watch}
            if ef["platform"] not in plats:
                return False, latency_s, f"{card.title_id} missing platform {ef['platform']} in {plats}", groq_calls
        if ef.get("genre"):
            genres = {g.strip() for g in (truth.get("genres") or "").split(",")}
            if ef["genre"] not in genres:
                return False, latency_s, f"{card.title_id} missing genre {ef['genre']} in {genres}", groq_calls
        if ef.get("any_genre"):
            genres = {g.strip() for g in (truth.get("genres") or "").split(",")}
            if not (genres & set(ef["any_genre"])):
                return False, latency_s, f"{card.title_id} genres={genres} none in {ef['any_genre']}", groq_calls
        if ef.get("exclude_title_id") and card.title_id == ef["exclude_title_id"]:
            return False, latency_s, f"anchor {ef['exclude_title_id']} was not excluded", groq_calls

    detail = f"{len(resp.picks)} picks" if resp.picks else resp.status
    return True, latency_s, detail, groq_calls


def test_eval_suite_has_exact_20_cases_breakdown() -> None:
    """Verify evals/questions.yaml contains exact 8+5+2+5 = 20 cases."""
    cases = _load_eval_cases()
    assert len(cases) == 20
    cats = [c["category"] for c in cases]
    assert cats.count("vibe") == 8
    assert cats.count("filter") == 5
    assert cats.count("more_like") == 2
    assert sum(1 for c in cats if c.startswith("guardrail_")) == 5


@pytest.mark.live_groq
def test_phase8_20_question_eval_suite() -> None:
    """Execute all 20 eval cases against live Groq + ChromaDB + SQLite and print pass/fail table."""
    if not config.GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not configured")

    # Pre-warm singletons so warm latency is measured accurately
    get_cached_embedder()
    get_cached_collection()
    catalog = _load_catalog_lookup()
    cases = _load_eval_cases()

    rows: list[tuple[str, str, str, bool, float, str]] = []
    failures: list[str] = []

    print("\n| ID | Category | Query | Result | Latency (s) | Details |")
    print("|---|---|---|---|---|---|")
    for case in cases:
        ok, latency_s, detail, groq_calls = validate_case(case, catalog)
        status_str = "PASS" if ok else "FAIL"
        rows.append((case["id"], case["category"], case["query"], ok, latency_s, detail))
        print(
            f"| {case['id']} | {case['category']} | {case['query']} | {status_str} | {latency_s:.2f} | {detail} |"
        )
        if not ok:
            failures.append(f"{case['id']}: {detail}")
        if groq_calls > 0:
            time.sleep(17)

    assert not failures, f"Failed eval cases: {failures}"
