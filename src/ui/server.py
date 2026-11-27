"""FastAPI web server serving the Movie Buddy chat UI and POST /api/chat."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src import config
from src.groq_client import GroqServiceError
from src.responder import RecommendationResponse, recommend
from src.retriever import get_cached_collection, is_embedder_loaded

STATIC_DIR: Path = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH: Path = STATIC_DIR / "index.html"


class ChatMessage(BaseModel):
    """Single conversation turn passed as optional context."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Incoming payload for POST /api/chat."""

    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Bind immediately on 0.0.0.0:$PORT; load FastEmbed lazily on first request."""
    yield


app = FastAPI(
    title="Movie Buddy",
    description="Vibe-based movie and series finder for India",
    lifespan=lifespan,
)


@app.get("/", response_class=FileResponse)
async def serve_index() -> FileResponse:
    """Serve the single-page Movie Buddy chatbot UI."""
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found.")
    return FileResponse(INDEX_HTML_PATH)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Readiness probe verifying SQLite, ChromaDB, and Groq configuration."""
    sqlite_ok = config.SQLITE_DB_PATH.exists()
    chroma_ok = config.CHROMA_DIR.exists()
    collection_count = 0
    if chroma_ok:
        try:
            collection_count = get_cached_collection().count()
        except Exception:
            chroma_ok = False
    cleaned_key = config.clean_env_secret(config.GROQ_API_KEY)
    groq_configured = bool(
        cleaned_key and cleaned_key != "your_groq_api_key_here"
    )
    status = "ok" if (sqlite_ok and chroma_ok and collection_count > 0) else "degraded"
    return {
        "status": status,
        "sqlite_ready": sqlite_ok,
        "chroma_ready": chroma_ok,
        "titles_indexed": collection_count,
        "embedder_loaded": is_embedder_loaded(),
        "groq_configured": groq_configured,
        "groq_key_format_ok": cleaned_key.startswith("gsk_"),
        "model": config.GROQ_MODEL,
    }


@app.post("/api/chat", response_model=RecommendationResponse)
async def chat_endpoint(payload: ChatRequest) -> JSONResponse | RecommendationResponse:
    """Run Movie Buddy RAG pipeline and return grounded picks or clear errors."""
    cleaned = payload.message.strip()
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Please enter a mood, plot, language, or title to search.",
        )

    history_dicts = [
        {"role": m.role, "content": m.content}
        for m in payload.history[-6:]
        if m.content.strip()
    ]

    try:
        result = await asyncio.to_thread(
            recommend,
            cleaned,
            history_dicts or None,
        )
        return result
    except GroqServiceError as exc:
        http_status = exc.status_code if exc.status_code in (401, 429, 502, 503) else 502
        err_resp = RecommendationResponse(
            status="error",
            message=exc.user_message,
            picks=[],
        )
        return JSONResponse(status_code=http_status, content=err_resp.model_dump())


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.ui.server:app", host="0.0.0.0", port=port, workers=1)
