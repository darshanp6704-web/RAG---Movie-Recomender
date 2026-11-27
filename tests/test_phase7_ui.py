"""Phase 7 UI server and single-page Stitch chatbot tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src import config
from src.groq_client import GroqServiceError
from src.responder import PickCard, RecommendationResponse
from src.retriever import WatchOption
from src.ui.server import app


def test_index_html_contains_stitch_theme_prompts_footer_and_display_none_modal() -> None:
    """Verify Stitch single-page UI elements, colors, 4 prompts, footer, and display:none modal."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Stitch Obsidian Cinema theme colors & typography
    assert "#111318" in html
    assert "#ffc174" in html
    assert "#f59e0b" in html
    assert "Plus Jakarta Sans" in html

    # Welcome, 4 example prompts, mandatory verbatim footer, and POST /api/chat wiring
    assert "Namaste! I'm Movie Buddy" in html
    assert html.count('class="example-prompt') == 4
    assert config.TMDB_JUSTWATCH_FOOTER in html
    assert "/api/chat" in html

    # Closed modal uses display: none so it never blocks clicks
    assert 'id="filter-modal" style="display: none;"' in html


def test_health_endpoint() -> None:
    """Verify /health reports ready SQLite and ChromaDB state."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["sqlite_ready"] is True
    assert data["chroma_ready"] is True
    assert data["titles_indexed"] == 3000


def test_chat_endpoint_rejects_blank_input() -> None:
    """EC-UI-05: Blank/whitespace message returns 422 validation error."""
    client = TestClient(app)
    resp = client.post("/api/chat", json={"message": "   "})
    assert resp.status_code == 422


def test_chat_endpoint_guardrail_and_card_payload() -> None:
    """Verify /api/chat blocks piracy and returns legal Indian watch options with as_of."""
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"message": "give me a torrent link to download Kantara free"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "blocked"
    assert data["violation_type"] == "piracy"
    assert len(data["picks"]) >= 1
    assert all(w["as_of"] for w in data["picks"][0]["where_to_watch"])
    assert data["footer"] == config.TMDB_JUSTWATCH_FOOTER


def test_chat_endpoint_surfaces_groq_errors_clearly(monkeypatch) -> None:
    """Verify GroqServiceError returns clean JSON error (no silent fallback)."""
    def fail_recommend(*args, **kwargs):
        raise GroqServiceError("Groq rate limit reached (HTTP 429).", status_code=429)

    monkeypatch.setattr("src.ui.server.recommend", fail_recommend)
    client = TestClient(app)
    resp = client.post("/api/chat", json={"message": "funny Hindi movie"})
    assert resp.status_code == 429
    data = resp.json()
    assert data["status"] == "error"
    assert "429" in data["message"]
    assert data["picks"] == []


def test_chat_endpoint_serializes_mood_and_watch_fields(monkeypatch) -> None:
    """Verify PickCard mood chips, where_to_watch, and as_of serialize cleanly."""
    mock_resp = RecommendationResponse(
        status="ok",
        message="Here is a great pick.",
        picks=[
            PickCard(
                title_id="m_191714",
                title="The Lunchbox",
                release_year=2013,
                language="Hindi",
                type="Movie",
                runtime_min=104,
                rating=7.4,
                certification_in="U",
                genres="Drama, Romance",
                mood="Romantic, Thought provoking, Comforting",
                hook="Can you fall in love with someone you haven't met?",
                why_it_matches="A gentle, heartwarming Hindi film under 2 hours.",
                where_to_watch=[
                    WatchOption(
                        platform="JioHotstar",
                        access_type="Free with ads",
                        region="IN",
                        as_of="2026-11-21",
                    )
                ],
                tmdb_url="https://www.themoviedb.org/movie/191714",
            )
        ],
    )
    monkeypatch.setattr("src.ui.server.recommend", lambda *a, **kw: mock_resp)
    client = TestClient(app)
    resp = client.post("/api/chat", json={"message": "light Hindi film"})
    assert resp.status_code == 200
    card = resp.json()["picks"][0]
    assert card["mood"] == "Romantic, Thought provoking, Comforting"
    assert card["where_to_watch"][0]["as_of"] == "2026-11-21"
