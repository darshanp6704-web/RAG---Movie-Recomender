"""Phase 5 tests for Groq parsing, grounded generation, logging, and strict error propagation."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import groq
import httpx
import pytest

from src import config
from src.groq_client import (
    GroqServiceError,
    call_groq_json,
    is_reasoning_model,
    resolve_max_tokens,
)
from src.query_parser import parse_query_with_groq
from src.responder import generate_recommendation_response, recommend
from src.retriever import QueryFilters, QuerySpec, retrieve


def test_reasoning_model_enforces_min_1024_max_tokens() -> None:
    """Verify reasoning models always receive at least 1024 max_tokens."""
    assert is_reasoning_model("openai/gpt-oss-20b") is True
    assert is_reasoning_model("qwen/qwen3.8-27b") is True
    assert resolve_max_tokens("openai/gpt-oss-20b", 512) == 1024
    assert resolve_max_tokens("openai/gpt-oss-20b", 1536) == 1536


def test_groq_error_raises_clearly_with_no_silent_fallback(caplog) -> None:
    """Verify Groq failures surface GroqServiceError and log status instead of falling back."""
    class FailingCompletions:
        @property
        def with_raw_response(self):
            class Raw:
                def create(self, **kwargs):
                    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
                    resp = httpx.Response(429, request=req)
                    raise groq.RateLimitError("rate limited", response=resp, body=None)
            return Raw()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    with caplog.at_level(logging.ERROR, logger="movie_buddy.groq"):
        with pytest.raises(GroqServiceError) as exc_info:
            parse_query_with_groq("feel good Hindi movie", client=fake_client)

    assert exc_info.value.status_code == 429
    assert "429" in exc_info.value.user_message
    assert any("status=429" in rec.message for rec in caplog.records)


def test_missing_api_key_raises_clearly() -> None:
    """Verify missing GROQ_API_KEY fails loudly instead of silently using offline fallback."""
    orig = config.GROQ_API_KEY
    config.GROQ_API_KEY = ""
    try:
        with pytest.raises(GroqServiceError) as exc_info:
            call_groq_json(
                purpose="test",
                system_prompt="sys",
                user_prompt="usr",
            )
        assert "GROQ_API_KEY" in exc_info.value.user_message
    finally:
        config.GROQ_API_KEY = orig


def test_responder_rejects_unretrieved_title_id() -> None:
    """Verify responder rejects LLM output referencing any title_id outside the <= 5 picks."""
    ret = retrieve(
        QuerySpec(
            raw_query="Hindi comedy",
            mode="filter_only",
            filters=QueryFilters(type="Movie", languages=["Hindi"], genres=["Comedy"]),
        )
    )
    assert len(ret.picks) == 5

    class HallucinatingCompletions:
        @property
        def with_raw_response(self):
            class Raw:
                def create(self, **kwargs):
                    payload = {
                        "intro": "Here are your picks.",
                        "reasons": {
                            "m_999999999": "Invented movie outside retrieved 5."
                        },
                    }
                    import json

                    parsed = SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
                        usage=SimpleNamespace(
                            prompt_tokens=100,
                            completion_tokens=40,
                            total_tokens=140,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
                        ),
                    )
                    return SimpleNamespace(status_code=200, parse=lambda: parsed)
            return Raw()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=HallucinatingCompletions()))
    with pytest.raises(GroqServiceError) as exc_info:
        generate_recommendation_response(
            user_query="Hindi comedy",
            picks=ret.picks,
            query_spec=ret.query_spec,
            status="ok",
            client=fake_client,
        )
    assert "unretrieved title IDs" in exc_info.value.user_message


@pytest.mark.live_groq
def test_live_groq_end_to_end_logging_and_grounding(caplog) -> None:
    """Live Groq integration test verifying status code, token logs, and grounded picks."""
    if not config.GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not configured")

    with caplog.at_level(logging.INFO, logger="movie_buddy.groq"):
        resp = recommend("light Kannada or Hindi film under 2 hours, free to watch")

    assert resp.status == "ok"
    assert 1 <= len(resp.picks) <= 5
    assert len(resp.groq_logs) == 2
    for log in resp.groq_logs:
        assert log.status_code == 200
        assert log.total_tokens > 0
        assert log.prompt_tokens > 0
        assert log.completion_tokens > 0

    for card in resp.picks:
        assert card.language in ("Kannada", "Hindi")
        assert card.type == "Movie"
        assert card.runtime_min is not None and card.runtime_min < 120
        assert card.why_it_matches
        assert card.tmdb_url
        assert all(w.as_of for w in card.where_to_watch)

    groq_info_logs = [r.message for r in caplog.records if "status=200" in r.message]
    assert len(groq_info_logs) >= 2
